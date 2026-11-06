// Oazy phone engine: runs our own models (Coral, Bee_v2_0.2B, Bee3, Frog) on a phone's CPU, fully offline.
//
// One file, no libraries. It reads the .oazy file that phone/export.py writes (8-bit or 4-bit weights, the SmolLM2
// tokenizer, and a reference answer from PyTorch for the self-check), maps it into memory, and talks to the app over
// stdin / stdout, one line per message (see serve() at the bottom).
//
// The maths is a copy of the PyTorch models' reading path (bee/model.py, memtrix/model.py, coral/model.py):
//   attention layer  RMSNorm -> q k v gate -> QK-norm -> RoPE (Coral) -> causal attention with grouped key/value
//                    heads -> times sigmoid(gate) -> out
//   memory layer     RMSNorm -> qkv -> 4-word causal convolution -> SiLU -> unit q, k -> the delta rule word by word
//                    (fade, guess, surprise, learn, recall) -> RMSNorm x SiLU(gate) -> out
//   thinking part    SwiGLU MLP, plus (Coral) a sigmoid router that picks the top 2 of 32 experts per word
//   head             the word table again (tied), soft-capped scores
// Weights are int8 (or int4 for the big MLP/expert grids) in groups of 32 with one float scale per group; the
// activations are rounded to int8 the same way just before each matrix product, so the inner loop is integer dot
// products (ARM's sdot where the chip has it). Everything else is float32.
//
// Build (phone): phone/build.sh. Build (PC, for the tests): c++ -O3 -std=c++17 -pthread phone/engine/oazy.cpp

#include <atomic>
#include <cstdarg>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <algorithm>

#if defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#endif

#include "unicode_tables.h"

using std::string;
using std::vector;

static void die(const char* fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stdout, "error\t");
    vfprintf(stdout, fmt, ap);
    fprintf(stdout, "\n");
    fflush(stdout);
    va_end(ap);
    exit(1);
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ====================================================================================== the thread pool
// The main thread and n-1 workers share each matrix product. Workers spin briefly between jobs (a token's ~300
// products follow each other within microseconds), then sleep, so an idle app costs nothing.
struct Pool {
    int n = 1;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> busy{0};
    std::atomic<int> next{0};
    int total = 0, chunk = 1;
    std::function<void(int, int)> job;
    std::mutex mu;
    std::condition_variable cv;
    bool quit = false;

    void start(int nt) {
        n = std::max(1, nt);
        for (int i = 1; i < n; i++) std::thread([this] { worker(); }).detach();   // they live as long as the process
    }
    void work() {
        for (;;) {
            int s = next.fetch_add(chunk);
            if (s >= total) break;
            job(s, std::min(total, s + chunk));
        }
    }
    void worker() {
        uint64_t seen = 0;
        for (;;) {
            int spins = 0;
            while (gen.load(std::memory_order_acquire) == seen) {
                if (++spins > 200000) {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return gen.load() != seen || quit; });
                }
                if (quit) return;
            }
            seen = gen.load();
            work();
            busy.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    // run fn(start, end) over [0, total) in pieces of `chunk`
    void run(int tot, int ch, std::function<void(int, int)> fn) {
        if (n == 1 || tot <= ch) { fn(0, tot); return; }
        job = std::move(fn);
        total = tot;
        chunk = ch;
        next.store(0);
        busy.store(n - 1);
        {
            std::lock_guard<std::mutex> lk(mu);
            gen.fetch_add(1, std::memory_order_acq_rel);
        }
        cv.notify_all();
        work();
        while (busy.load(std::memory_order_acquire) > 0) {}
    }
};
static Pool& POOL = *new Pool;     // never destroyed: its sleeping workers must not see the condition variable go away at exit

// Put the worker threads on the fastest cores (phones mix big and little ones).
static void pin_to_fast_cores(int nt) {
#if defined(__linux__)
    vector<std::pair<long, int>> f;
    for (int c = 0; c < 64; c++) {
        char p[128];
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        FILE* fp = fopen(p, "r");
        if (!fp) continue;
        long v = 0;
        if (fscanf(fp, "%ld", &v) == 1) f.push_back({v, c});
        fclose(fp);
    }
    if ((int)f.size() < 2) return;
    std::sort(f.begin(), f.end(), [](auto a, auto b) { return a.first > b.first; });
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < std::min(nt, (int)f.size()); i++) CPU_SET(f[i].second, &set);
    sched_setaffinity(0, sizeof set, &set);     // inherited by the threads made after this
#else
    (void)nt;
#endif
}

// ====================================================================================== weights
enum { T_F32 = 0, T_Q8 = 1, T_Q4 = 2 };

struct Mat {
    int type = -1, rows = 0, cols = 0;
    const float* f = nullptr;      // T_F32: rows x cols
    const int8_t* q = nullptr;     // T_Q8: rows x cols
    const uint8_t* q4 = nullptr;   // T_Q4: rows x cols/2 (two 4-bit values a byte, +8)
    const float* s = nullptr;      // scales: rows x cols/32
    Mat sub(int r0, int nr) const {   // rows [r0, r0 + nr)
        Mat m = *this;
        m.rows = nr;
        if (f) m.f = f + (size_t)r0 * cols;
        if (q) m.q = q + (size_t)r0 * cols;
        if (q4) m.q4 = q4 + (size_t)r0 * cols / 2;
        if (s) m.s = s + (size_t)r0 * (cols / 32);
        return m;
    }
};

struct File {
    const uint8_t* base = nullptr;
    size_t size = 0;
    std::unordered_map<string, string> cfg;
    std::unordered_map<string, Mat> mats;
    std::unordered_map<string, std::pair<const uint8_t*, size_t>> blobs;

    void open(const char* path) {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) die("cannot open %s: %s", path, strerror(errno));
        struct stat st;
        fstat(fd, &st);
        size = st.st_size;
        void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) die("cannot map %s: %s", path, strerror(errno));
        close(fd);
        base = (const uint8_t*)p;
        if (size < 16 || memcmp(base, "OAZYPHN1", 8) != 0) die("%s is not an Oazy phone model", path);
        uint64_t hl;
        memcpy(&hl, base + 8, 8);
        string head((const char*)base + 16, hl);
        const size_t data0 = (16 + hl + 63) / 64 * 64;     // the data starts at the first 64-byte boundary after the header
        size_t i = 0;
        while (i < head.size()) {
            size_t e = head.find('\n', i);
            if (e == string::npos) e = head.size();
            string line = head.substr(i, e - i);
            i = e + 1;
            if (line.empty()) continue;
            if (line.rfind("tensor ", 0) == 0) {
                char name[256];
                int type, rows, cols;
                unsigned long long off, nb;
                if (sscanf(line.c_str(), "tensor %255s %d %d %d %llu %llu", name, &type, &rows, &cols, &off, &nb) != 6)
                    die("bad tensor line: %s", line.c_str());
                if (data0 + off + nb > size) die("the model file is cut short (tensor %s); copy it again", name);
                Mat m;
                m.type = type;
                m.rows = rows;
                m.cols = cols;
                const uint8_t* d = base + data0 + off;
                if (type == T_F32) m.f = (const float*)d;
                else if (type == T_Q8) { m.q = (const int8_t*)d; m.s = (const float*)(d + (size_t)rows * cols); }
                else if (type == T_Q4) { m.q4 = d; m.s = (const float*)(d + (size_t)rows * cols / 2); }
                else if (type == 9) { blobs[name] = {d, (size_t)nb}; continue; }
                mats[name] = m;
            } else {
                size_t sp = line.find(' ');
                if (sp == string::npos) cfg[line] = "";
                else cfg[line.substr(0, sp)] = line.substr(sp + 1);
            }
        }
    }
    bool has(const string& k) const { return cfg.count(k); }
    string get(const string& k, const string& d = "") const { auto it = cfg.find(k); return it == cfg.end() ? d : it->second; }
    int geti(const string& k, int d = 0) const { auto it = cfg.find(k); return it == cfg.end() ? d : atoi(it->second.c_str()); }
    double getf(const string& k, double d = 0) const { auto it = cfg.find(k); return it == cfg.end() ? d : atof(it->second.c_str()); }
    const Mat& mat(const string& k) const {
        auto it = mats.find(k);
        if (it == mats.end()) die("the model file has no tensor %s", k.c_str());
        return it->second;
    }
    const Mat* opt(const string& k) const { auto it = mats.find(k); return it == mats.end() ? nullptr : &it->second; }
    std::pair<const uint8_t*, size_t> blob(const string& k) const {
        auto it = blobs.find(k);
        return it == blobs.end() ? std::pair<const uint8_t*, size_t>{nullptr, 0} : it->second;
    }
};

// ====================================================================================== the integer dot products
// Activations: int8 in groups of 32, one float scale per group (round(x / (max|x| / 127))).
struct QAct {
    vector<int8_t> q;
    vector<float> s;
    int n = 0, cols = 0;
    void resize(int nn, int c) { n = nn; cols = c; q.resize((size_t)nn * c); s.resize((size_t)nn * c / 32); }
    const int8_t* row(int i) const { return q.data() + (size_t)i * cols; }
    const float* srow(int i) const { return s.data() + (size_t)i * cols / 32; }
};

static void quantize(const float* x, int n, int cols, QAct& out) {
    out.resize(n, cols);
    for (int i = 0; i < n; i++) {
        const float* xr = x + (size_t)i * cols;
        int8_t* qr = out.q.data() + (size_t)i * cols;
        float* sr = out.s.data() + (size_t)i * cols / 32;
        for (int g = 0; g < cols / 32; g++) {
            const float* xg = xr + g * 32;
            float m = 0;
            for (int j = 0; j < 32; j++) m = std::max(m, std::fabs(xg[j]));
            float sc = m / 127.f, inv = m > 0 ? 127.f / m : 0.f;
            sr[g] = sc;
            for (int j = 0; j < 32; j++) qr[g * 32 + j] = (int8_t)lrintf(xg[j] * inv);
        }
    }
}

// dot of one weight row with up to 4 activation rows: out[b] = sum_g (w_g . x_bg) * ws_g * xs_bg
typedef void (*DotQ8)(const int8_t* w, const float* ws, const QAct& x, int b0, int nb, int groups, float* out);
typedef void (*DotQ4)(const uint8_t* w, const float* ws, const QAct& x, int b0, int nb, int groups, float* out);

static void dot_q8_plain(const int8_t* w, const float* ws, const QAct& x, int b0, int nb, int groups, float* out) {
    for (int b = 0; b < nb; b++) {
        const int8_t* xq = x.row(b0 + b);
        const float* xs = x.srow(b0 + b);
        float acc = 0;
        for (int g = 0; g < groups; g++) {
            int s = 0;
            for (int j = 0; j < 32; j++) s += (int)w[g * 32 + j] * (int)xq[g * 32 + j];
            acc += (float)s * ws[g] * xs[g];
        }
        out[b] = acc;
    }
}
static void dot_q4_plain(const uint8_t* w, const float* ws, const QAct& x, int b0, int nb, int groups, float* out) {
    for (int b = 0; b < nb; b++) {
        const int8_t* xq = x.row(b0 + b);
        const float* xs = x.srow(b0 + b);
        float acc = 0;
        for (int g = 0; g < groups; g++) {
            int s = 0;
            const uint8_t* wg = w + g * 16;
            for (int j = 0; j < 16; j++) {     // low nibbles are values 0..15 of the group, high ones 16..31
                s += ((int)(wg[j] & 15) - 8) * (int)xq[g * 32 + j];
                s += ((int)(wg[j] >> 4) - 8) * (int)xq[g * 32 + 16 + j];
            }
            acc += (float)s * ws[g] * xs[g];
        }
        out[b] = acc;
    }
}

#if defined(__aarch64__)
// ARMv8.0 NEON: widening multiplies (every 64-bit phone).
static void dot_q8_neon(const int8_t* w, const float* ws, const QAct& x, int b0, int nb, int groups, float* out) {
    for (int b = 0; b < nb; b++) {
        const int8_t* xq = x.row(b0 + b);
        const float* xs = x.srow(b0 + b);
        float32x4_t acc = vdupq_n_f32(0);
        for (int g = 0; g < groups; g++) {
            int8x16_t w0 = vld1q_s8(w + g * 32), w1 = vld1q_s8(w + g * 32 + 16);
            int8x16_t x0 = vld1q_s8(xq + g * 32), x1 = vld1q_s8(xq + g * 32 + 16);
            int16x8_t p = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
            p = vmlal_s8(p, vget_high_s8(w0), vget_high_s8(x0));
            int16x8_t r = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
            r = vmlal_s8(r, vget_high_s8(w1), vget_high_s8(x1));
            int32x4_t s = vpaddlq_s16(p);
            s = vpadalq_s16(s, r);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(s), ws[g] * xs[g]);
        }
        out[b] = vaddvq_f32(acc);
    }
}
static void dot_q4_neon(const uint8_t* w, const float* ws, const QAct& x, int b0, int nb, int groups, float* out) {
    const uint8x16_t m4 = vdupq_n_u8(15);
    const int8x16_t e8 = vdupq_n_s8(8);
    for (int b = 0; b < nb; b++) {
        const int8_t* xq = x.row(b0 + b);
        const float* xs = x.srow(b0 + b);
        float32x4_t acc = vdupq_n_f32(0);
        for (int g = 0; g < groups; g++) {
            uint8x16_t raw = vld1q_u8(w + g * 16);
            int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, m4)), e8);
            int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), e8);
            int8x16_t x0 = vld1q_s8(xq + g * 32), x1 = vld1q_s8(xq + g * 32 + 16);
            int16x8_t p = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
            p = vmlal_s8(p, vget_high_s8(w0), vget_high_s8(x0));
            int16x8_t r = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
            r = vmlal_s8(r, vget_high_s8(w1), vget_high_s8(x1));
            int32x4_t s = vpaddlq_s16(p);
            s = vpadalq_s16(s, r);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(s), ws[g] * xs[g]);
        }
        out[b] = vaddvq_f32(acc);
    }
}
// ARMv8.2 dot-product instructions (most phones since 2019): 4x fewer instructions per group.
__attribute__((target("dotprod")))
static void dot_q8_sdot(const int8_t* w, const float* ws, const QAct& x, int b0, int nb, int groups, float* out) {
    int b = 0;
    for (; b + 2 <= nb; b += 2) {       // two activation rows share each weight load
        const int8_t* xa = x.row(b0 + b);
        const int8_t* xb = x.row(b0 + b + 1);
        const float* sa = x.srow(b0 + b);
        const float* sb = x.srow(b0 + b + 1);
        float32x4_t aa = vdupq_n_f32(0), ab = vdupq_n_f32(0);
        for (int g = 0; g < groups; g++) {
            int8x16_t w0 = vld1q_s8(w + g * 32), w1 = vld1q_s8(w + g * 32 + 16);
            int32x4_t pa = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(xa + g * 32)), w1, vld1q_s8(xa + g * 32 + 16));
            int32x4_t pb = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(xb + g * 32)), w1, vld1q_s8(xb + g * 32 + 16));
            aa = vmlaq_n_f32(aa, vcvtq_f32_s32(pa), ws[g] * sa[g]);
            ab = vmlaq_n_f32(ab, vcvtq_f32_s32(pb), ws[g] * sb[g]);
        }
        out[b] = vaddvq_f32(aa);
        out[b + 1] = vaddvq_f32(ab);
    }
    for (; b < nb; b++) {
        const int8_t* xq = x.row(b0 + b);
        const float* xs = x.srow(b0 + b);
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        int g = 0;
        for (; g + 2 <= groups; g += 2) {    // two independent chains keep the pipeline full
            int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w + g * 32), vld1q_s8(xq + g * 32)),
                                    vld1q_s8(w + g * 32 + 16), vld1q_s8(xq + g * 32 + 16));
            int32x4_t r = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w + g * 32 + 32), vld1q_s8(xq + g * 32 + 32)),
                                    vld1q_s8(w + g * 32 + 48), vld1q_s8(xq + g * 32 + 48));
            a0 = vmlaq_n_f32(a0, vcvtq_f32_s32(p), ws[g] * xs[g]);
            a1 = vmlaq_n_f32(a1, vcvtq_f32_s32(r), ws[g + 1] * xs[g + 1]);
        }
        for (; g < groups; g++) {
            int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w + g * 32), vld1q_s8(xq + g * 32)),
                                    vld1q_s8(w + g * 32 + 16), vld1q_s8(xq + g * 32 + 16));
            a0 = vmlaq_n_f32(a0, vcvtq_f32_s32(p), ws[g] * xs[g]);
        }
        out[b] = vaddvq_f32(vaddq_f32(a0, a1));
    }
}
__attribute__((target("dotprod")))
static void dot_q4_sdot(const uint8_t* w, const float* ws, const QAct& x, int b0, int nb, int groups, float* out) {
    const uint8x16_t m4 = vdupq_n_u8(15);
    const int8x16_t e8 = vdupq_n_s8(8);
    for (int b = 0; b < nb; b++) {
        const int8_t* xq = x.row(b0 + b);
        const float* xs = x.srow(b0 + b);
        float32x4_t acc = vdupq_n_f32(0);
        for (int g = 0; g < groups; g++) {
            uint8x16_t raw = vld1q_u8(w + g * 16);
            int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, m4)), e8);
            int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), e8);
            int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(xq + g * 32)), w1, vld1q_s8(xq + g * 32 + 16));
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), ws[g] * xs[g]);
        }
        out[b] = vaddvq_f32(acc);
    }
}
#endif

static DotQ8 DOT_Q8 = dot_q8_plain;
static DotQ4 DOT_Q4 = dot_q4_plain;
static const char* KERNEL = "plain C";

static void pick_kernels() {
#if defined(__aarch64__)
    if (getenv("OAZY_NO_SDOT") == nullptr && (getauxval(AT_HWCAP) & HWCAP_ASIMDDP)) {
        DOT_Q8 = dot_q8_sdot; DOT_Q4 = dot_q4_sdot; KERNEL = "ARM sdot";
    } else {
        DOT_Q8 = dot_q8_neon; DOT_Q4 = dot_q4_neon; KERNEL = "ARM NEON";
    }
#endif
}

// y (n x W.rows) = x (n x W.cols) @ W^T. x quantized for Q8/Q4, float for F32. ldy: row stride of y.
static void matmul(const Mat& W, const QAct& xq, const float* xf, int n, float* y, int ldy) {
    const int R = W.rows, C = W.cols, G = C / 32;
    int ch = std::max(8, R / (POOL.n * 8));
    ch = (ch + 7) & ~7;
    POOL.run(R, ch, [&](int r0, int r1) {
        float tmp[4];
        for (int r = r0; r < r1; r++) {
            for (int b0 = 0; b0 < n; b0 += 4) {
                int nb = std::min(4, n - b0);
                if (W.type == T_Q8) DOT_Q8(W.q + (size_t)r * C, W.s + (size_t)r * G, xq, b0, nb, G, tmp);
                else if (W.type == T_Q4) DOT_Q4(W.q4 + (size_t)r * C / 2, W.s + (size_t)r * G, xq, b0, nb, G, tmp);
                else {
                    for (int b = 0; b < nb; b++) {
                        const float* wr = W.f + (size_t)r * C;
                        const float* xr = xf + (size_t)(b0 + b) * C;
                        float a = 0;
                        for (int j = 0; j < C; j++) a += wr[j] * xr[j];
                        tmp[b] = a;
                    }
                }
                for (int b = 0; b < nb; b++) y[(size_t)(b0 + b) * ldy + r] = tmp[b];
            }
        }
    });
}

// ====================================================================================== small maths
static inline float silu(float x) { return x / (1.f + std::exp(-x)); }
static inline float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }
static inline float softplus(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }

static void rmsnorm(const float* x, const float* w, int d, float* y, float eps = 1e-6f) {
    double ss = 0;
    for (int i = 0; i < d; i++) ss += (double)x[i] * x[i];
    float r = 1.f / std::sqrt((float)(ss / d) + eps);
    for (int i = 0; i < d; i++) y[i] = x[i] * r * w[i];
}

static void dequant_row(const Mat& m, int r, float* out) {
    const int C = m.cols;
    if (m.type == T_F32) { memcpy(out, m.f + (size_t)r * C, C * sizeof(float)); return; }
    for (int g = 0; g < C / 32; g++) {
        float s = m.s[(size_t)r * (C / 32) + g];
        for (int j = 0; j < 32; j++) {
            int v;
            if (m.type == T_Q8) v = m.q[(size_t)r * C + g * 32 + j];
            else {
                uint8_t byte = m.q4[(size_t)r * C / 2 + g * 16 + (j & 15)];
                v = (j < 16 ? (byte & 15) : (byte >> 4)) - 8;
            }
            out[g * 32 + j] = v * s;
        }
    }
}

// ====================================================================================== the model
struct Layer {
    char kind = 'A';
    const float *n1 = nullptr, *n2 = nullptr;
    // attention
    Mat wqkvg, wo;
    const float *qn = nullptr, *kn = nullptr;
    // memory
    Mat wproj;                        // qkv | gate | trust | fade
    const float *bproj = nullptr;     // trust and fade biases (2H)
    const float *conv = nullptr;      // (ch, w)
    const float *A_log = nullptr, *dt_bias = nullptr, *onorm = nullptr;
    // thinking
    Mat wup, wdown;
    bool has_mlp = false;
    Mat router, eup, edown;
    const float* rbias = nullptr;
    bool has_moe = false;
    // state
    vector<float> kc, vc;             // attention: (ctx, KV * hd)
    vector<float> S, hist;            // memory: (H, K, V) and the last w - 1 words' channels
};

struct Model {
    File F;
    string name, kind, quant;
    int D, L, V, nvocab, eot;
    float softcap;
    int AH, AKV, hd, rope;
    float rope_base;
    int MH, MK, MV, conv;
    int E, topk, EH, mlp_hidden;
    int ctx;
    vector<Layer> layers;
    Mat embed;
    const float* norm;
    int pos = 0;
    vector<float> rope_cos, rope_sin;   // (ctx, hd / 2)

    // scratch
    int NB = 32;
    vector<float> x, h, big, big2, att, logits;
    QAct xq, xq2;

    void load(const char* path, int max_ctx) {
        F.open(path);
        name = F.get("name", "model");
        kind = F.get("kind");
        quant = F.get("quant");
        D = F.geti("d_model");
        L = F.geti("n_layer");
        V = F.geti("vocab_size");
        nvocab = F.geti("n_vocab", V);
        eot = F.geti("eot");
        softcap = (float)F.getf("softcap");
        AH = F.geti("attn_heads"); AKV = F.geti("attn_kv_heads"); hd = F.geti("attn_head_dim");
        rope = F.geti("rope"); rope_base = (float)F.getf("rope_base", 10000);
        MH = F.geti("n_head"); MK = F.geti("head_k"); MV = F.geti("head_v"); conv = F.geti("conv", 4);
        E = F.geti("n_expert"); topk = F.geti("top_k"); EH = F.geti("expert_hidden");
        mlp_hidden = F.geti("mlp_hidden");
        ctx = max_ctx > 0 ? max_ctx : F.geti("max_len", 2048);    // default: the length it was trained on
        string kinds = F.get("layers");
        if ((int)kinds.size() != L) die("layer list does not match n_layer");
        embed = F.mat("embed");
        norm = F.mat("norm").f;
        layers.resize(L);
        int maxw = 0;
        for (int i = 0; i < L; i++) {
            Layer& l = layers[i];
            string p = "L" + std::to_string(i) + ".";
            l.kind = kinds[i];
            l.n1 = F.mat(p + "n1").f;
            l.n2 = F.mat(p + "n2").f;
            if (l.kind == 'A') {
                l.wqkvg = F.mat(p + "wqkvg"); l.wo = F.mat(p + "wo");
                l.qn = F.mat(p + "qn").f; l.kn = F.mat(p + "kn").f;
                l.kc.assign((size_t)ctx * AKV * hd, 0.f);
                l.vc.assign((size_t)ctx * AKV * hd, 0.f);
                maxw = std::max(maxw, l.wqkvg.rows);
            } else {
                l.wproj = F.mat(p + "wproj"); l.wo = F.mat(p + "wo");
                l.bproj = F.mat(p + "bproj").f; l.conv = F.mat(p + "conv").f;
                l.A_log = F.mat(p + "A_log").f; l.dt_bias = F.mat(p + "dt_bias").f; l.onorm = F.mat(p + "onorm").f;
                int ch = MH * (2 * MK + MV);
                l.S.assign((size_t)MH * MK * MV, 0.f);
                l.hist.assign((size_t)(conv - 1) * ch, 0.f);
                maxw = std::max(maxw, l.wproj.rows);
            }
            if (F.opt(p + "wup")) {
                l.has_mlp = true; l.wup = F.mat(p + "wup"); l.wdown = F.mat(p + "wdown");
                maxw = std::max(maxw, l.wup.rows);
            }
            if (F.opt(p + "eup")) {
                l.has_moe = true; l.router = F.mat(p + "router"); l.eup = F.mat(p + "eup"); l.edown = F.mat(p + "edown");
                l.rbias = F.mat(p + "rbias").f;
                maxw = std::max(maxw, 2 * EH);
            }
        }
        maxw = std::max(maxw, D);
        x.assign((size_t)NB * D, 0); h.assign((size_t)NB * D, 0);
        big.assign((size_t)NB * maxw, 0); big2.assign((size_t)NB * std::max(maxw, D), 0);
        att.assign((size_t)NB * std::max({AH * hd, MH * MV, D, 1}), 0);
        logits.assign((size_t)V, 0);
        if (rope) {
            int half = hd / 2;
            rope_cos.resize((size_t)ctx * half); rope_sin.resize((size_t)ctx * half);
            for (int i = 0; i < half; i++) {
                // as torch: inv = 1 / base ** (arange(0, d, 2) / d), in float32
                float inv = 1.0f / std::pow(rope_base, (float)(2 * i) / (float)hd);
                for (int t = 0; t < ctx; t++) {
                    float a = (float)t * inv;
                    rope_cos[(size_t)t * half + i] = std::cos(a);
                    rope_sin[(size_t)t * half + i] = std::sin(a);
                }
            }
        }
    }

    void reset() {
        pos = 0;
        for (auto& l : layers) {
            std::fill(l.S.begin(), l.S.end(), 0.f);
            std::fill(l.hist.begin(), l.hist.end(), 0.f);
        }
    }

    // ---------------------------------------------------------------- one attention layer over n new words
    void attention(Layer& l, int n) {
        const int qd = AH * hd, kd = AKV * hd, W = l.wqkvg.rows, rep = AH / AKV, half = hd / 2;
        for (int i = 0; i < n; i++) rmsnorm(&x[(size_t)i * D], l.n1, D, &h[(size_t)i * D]);
        quantize(h.data(), n, D, xq);
        matmul(l.wqkvg, xq, h.data(), n, big.data(), W);
        for (int i = 0; i < n; i++) {
            float* r = &big[(size_t)i * W];
            float *q = r, *k = r + qd, *v = r + qd + kd;
            for (int a = 0; a < AH; a++) rmsnorm(q + a * hd, l.qn, hd, q + a * hd);
            for (int a = 0; a < AKV; a++) rmsnorm(k + a * hd, l.kn, hd, k + a * hd);
            int t = pos + i;
            if (rope) {
                const float* c = &rope_cos[(size_t)t * half];
                const float* s = &rope_sin[(size_t)t * half];
                auto rot = [&](float* u) {
                    for (int j = 0; j < half; j++) {
                        float x1 = u[j], x2 = u[j + half];
                        u[j] = x1 * c[j] - x2 * s[j];
                        u[j + half] = x1 * s[j] + x2 * c[j];
                    }
                };
                for (int a = 0; a < AH; a++) rot(q + a * hd);
                for (int a = 0; a < AKV; a++) rot(k + a * hd);
            }
            memcpy(&l.kc[(size_t)t * kd], k, kd * sizeof(float));
            memcpy(&l.vc[(size_t)t * kd], v, kd * sizeof(float));
        }
        const float scale = 1.f / std::sqrt((float)hd);
        POOL.run(n * AH, 1, [&](int j0, int j1) {
            vector<float> sc(pos + n);
            for (int j = j0; j < j1; j++) {
                int i = j / AH, a = j % AH, kv = a / rep, T = pos + i + 1;
                const float* q = &big[(size_t)i * W + a * hd];
                float m = -1e30f;
                for (int t = 0; t < T; t++) {
                    const float* k = &l.kc[(size_t)t * kd + kv * hd];
                    float s = 0;
                    for (int u = 0; u < hd; u++) s += q[u] * k[u];
                    sc[t] = s * scale;
                    m = std::max(m, sc[t]);
                }
                float sum = 0;
                for (int t = 0; t < T; t++) { sc[t] = std::exp(sc[t] - m); sum += sc[t]; }
                float* o = &att[(size_t)i * qd + a * hd];
                for (int u = 0; u < hd; u++) o[u] = 0;
                for (int t = 0; t < T; t++) {
                    const float* v = &l.vc[(size_t)t * kd + kv * hd];
                    float p = sc[t] / sum;
                    for (int u = 0; u < hd; u++) o[u] += p * v[u];
                }
                const float* g = &big[(size_t)i * W + qd + 2 * kd + a * hd];
                for (int u = 0; u < hd; u++) o[u] *= sigm(g[u]);
            }
        });
        quantize(att.data(), n, qd, xq);
        matmul(l.wo, xq, att.data(), n, big2.data(), D);
        for (size_t i = 0; i < (size_t)n * D; i++) x[i] += big2[i];
    }

    // ---------------------------------------------------------------- one memory (Memtrix) layer over n new words
    void memory(Layer& l, int n) {
        const int H = MH, K = MK, Vd = MV, ch = H * (2 * K + Vd), W = l.wproj.rows, w = conv;
        for (int i = 0; i < n; i++) rmsnorm(&x[(size_t)i * D], l.n1, D, &h[(size_t)i * D]);
        quantize(h.data(), n, D, xq);
        matmul(l.wproj, xq, h.data(), n, big.data(), W);
        const float kscale = 1.f / std::sqrt((float)K);
        vector<float> z(ch), qv(H * K), kv(H * K), tr(H), lf(H);
        for (int i = 0; i < n; i++) {
            float* r = &big[(size_t)i * W];
            // the 4-word causal convolution over the qkv channels (hist[0] = the previous word)
            for (int c = 0; c < ch; c++) {
                float s = l.conv[c * w + (w - 1)] * r[c];
                for (int j = 1; j < w; j++) s += l.conv[c * w + (w - 1 - j)] * l.hist[(size_t)(j - 1) * ch + c];
                z[c] = silu(s);
            }
            for (int j = w - 2; j >= 1; j--) memcpy(&l.hist[(size_t)j * ch], &l.hist[(size_t)(j - 1) * ch], ch * sizeof(float));
            if (w > 1) memcpy(&l.hist[0], r, ch * sizeof(float));
            const float* zq = z.data();
            const float* zk = z.data() + H * K;
            const float* zv = z.data() + 2 * H * K;
            for (int a = 0; a < H; a++) {
                double nq = 0, nk = 0;
                for (int u = 0; u < K; u++) { nq += (double)zq[a * K + u] * zq[a * K + u]; nk += (double)zk[a * K + u] * zk[a * K + u]; }
                float iq = 1.f / std::max((float)std::sqrt(nq), 1e-12f), ik = 1.f / std::max((float)std::sqrt(nk), 1e-12f);
                for (int u = 0; u < K; u++) { qv[a * K + u] = zq[a * K + u] * iq; kv[a * K + u] = zk[a * K + u] * ik; }
                tr[a] = sigm(r[ch + H * Vd + a] + l.bproj[a]);
                lf[a] = -std::exp(l.A_log[a]) * softplus(r[ch + H * Vd + H + a] + l.bproj[H + a] + l.dt_bias[a]);
            }
            float* o = &att[(size_t)i * H * Vd];
            POOL.run(H, 1, [&](int a0, int a1) {
                vector<float> guess(Vd);
                for (int a = a0; a < a1; a++) {
                    float* S = &l.S[(size_t)a * K * Vd];
                    const float fade = std::exp(lf[a]);
                    const float* k = &kv[a * K];
                    const float* q = &qv[a * K];
                    for (int u = 0; u < Vd; u++) guess[u] = 0;
                    for (int r2 = 0; r2 < K; r2++) {
                        float* Sr = S + (size_t)r2 * Vd;
                        for (int u = 0; u < Vd; u++) { Sr[u] *= fade; guess[u] += k[r2] * Sr[u]; }
                    }
                    for (int u = 0; u < Vd; u++) guess[u] = tr[a] * (zv[a * Vd + u] - guess[u]);   // trust x surprise
                    float* oa = o + a * Vd;
                    for (int u = 0; u < Vd; u++) oa[u] = 0;
                    for (int r2 = 0; r2 < K; r2++) {
                        float* Sr = S + (size_t)r2 * Vd;
                        float kr = k[r2], qr = q[r2] * kscale;
                        for (int u = 0; u < Vd; u++) { Sr[u] += kr * guess[u]; oa[u] += qr * Sr[u]; }
                    }
                    rmsnorm(oa, l.onorm, Vd, oa);
                    const float* g = r + ch + a * Vd;
                    for (int u = 0; u < Vd; u++) oa[u] *= silu(g[u]);
                }
            });
        }
        quantize(att.data(), n, H * Vd, xq);
        matmul(l.wo, xq, att.data(), n, big2.data(), D);
        for (size_t i = 0; i < (size_t)n * D; i++) x[i] += big2[i];
    }

    // ---------------------------------------------------------------- the thinking part: MLP + experts
    void think(Layer& l, int n) {
        for (int i = 0; i < n; i++) rmsnorm(&x[(size_t)i * D], l.n2, D, &h[(size_t)i * D]);
        quantize(h.data(), n, D, xq2);          // shared by the MLP and the experts
        vector<float> add((size_t)n * D, 0.f);
        if (l.has_mlp) {
            const int Hd = l.wup.rows / 2;
            matmul(l.wup, xq2, h.data(), n, big.data(), 2 * Hd);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < Hd; j++) big2[(size_t)i * Hd + j] = silu(big[(size_t)i * 2 * Hd + j]) * big[(size_t)i * 2 * Hd + Hd + j];
            quantize(big2.data(), n, Hd, xq);
            matmul(l.wdown, xq, big2.data(), n, big.data(), D);
            for (size_t i = 0; i < (size_t)n * D; i++) add[i] += big[i];
        }
        if (l.has_moe) {
            // the router in float32: sigmoid scores; the balancing bias steers the choice only
            vector<float> lg((size_t)n * E);
            QAct none;
            matmul(l.router, none, h.data(), n, lg.data(), E);
            vector<vector<std::pair<int, float>>> per(E);
            for (int i = 0; i < n; i++) {
                vector<float> sc(E), sel(E);
                for (int e = 0; e < E; e++) { sc[e] = sigm(lg[(size_t)i * E + e]); sel[e] = sc[e] + l.rbias[e]; }
                vector<int> idx(E);
                for (int e = 0; e < E; e++) idx[e] = e;
                std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int b) { return sel[a] > sel[b]; });
                float tot = 0;
                for (int j = 0; j < topk; j++) tot += sc[idx[j]];
                for (int j = 0; j < topk; j++) per[idx[j]].push_back({i, sc[idx[j]] / (tot + 1e-9f)});
            }
            QAct g;
            vector<float> hf;
            for (int e = 0; e < E; e++) {
                if (per[e].empty()) continue;
                int m = (int)per[e].size();
                g.resize(m, D);
                hf.resize((size_t)m * D);
                for (int j = 0; j < m; j++) {        // gather this expert's words (already rounded to int8)
                    memcpy(g.q.data() + (size_t)j * D, xq2.row(per[e][j].first), D);
                    memcpy(g.s.data() + (size_t)j * D / 32, xq2.srow(per[e][j].first), D / 32 * sizeof(float));
                    memcpy(hf.data() + (size_t)j * D, &h[(size_t)per[e][j].first * D], D * sizeof(float));
                }
                matmul(l.eup.sub(e * 2 * EH, 2 * EH), g, hf.data(), m, big.data(), 2 * EH);
                for (int j = 0; j < m; j++)
                    for (int u = 0; u < EH; u++) big2[(size_t)j * EH + u] = silu(big[(size_t)j * 2 * EH + u]) * big[(size_t)j * 2 * EH + EH + u];
                quantize(big2.data(), m, EH, xq);
                matmul(l.edown.sub(e * D, D), xq, big2.data(), m, big.data(), D);
                for (int j = 0; j < m; j++) {
                    float wgt = per[e][j].second;
                    float* dst = &add[(size_t)per[e][j].first * D];
                    for (int u = 0; u < D; u++) dst[u] += wgt * big[(size_t)j * D + u];
                }
            }
        }
        for (size_t i = 0; i < (size_t)n * D; i++) x[i] += add[i];
    }

    // Read n words (n <= NB). all_logits: the scores of every word into out (n x V), else of the last only.
    void forward(const int* tok, int n, float* out, bool all_logits) {
        if (pos + n > ctx) die("context full");
        vector<float> row(D);
        for (int i = 0; i < n; i++) dequant_row(embed, tok[i], &x[(size_t)i * D]);
        for (auto& l : layers) {
            if (l.kind == 'A') attention(l, n); else memory(l, n);
            think(l, n);
        }
        int first = all_logits ? 0 : n - 1, m = n - first;
        for (int i = 0; i < m; i++) rmsnorm(&x[(size_t)(first + i) * D], norm, D, &h[(size_t)i * D]);
        quantize(h.data(), m, D, xq);
        matmul(embed, xq, h.data(), m, out, V);
        if (softcap > 0)
            for (size_t i = 0; i < (size_t)m * V; i++) out[i] = softcap * std::tanh(out[i] / softcap);
        pos += n;
    }
};

// ====================================================================================== the tokenizer (byte-level BPE)
// SmolLM2's: split off each digit, then GPT-2's pattern ('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
// ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+), then merge the bytes of each piece by the merge list.
static bool in_ranges(uint32_t c, const uint32_t (*r)[2], int n) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        if (c < r[m][0]) hi = m - 1;
        else if (c > r[m][1]) lo = m + 1;
        else return true;
    }
    return false;
}
static bool is_L(uint32_t c) { return in_ranges(c, UNI_L, UNI_L_N); }
static bool is_N(uint32_t c) { return in_ranges(c, UNI_N, UNI_N_N); }
static bool is_S(uint32_t c) {
    return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

static vector<uint32_t> utf8_decode(const string& s, vector<int>& starts) {
    vector<uint32_t> cps;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        uint32_t cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 6) { cp = c & 31; len = 2; }
        else if ((c >> 4) == 14) { cp = c & 15; len = 3; }
        else if ((c >> 3) == 30) { cp = c & 7; len = 4; }
        else { cp = 0xFFFD; len = 1; }
        if (i + len > s.size()) { cp = 0xFFFD; len = 1; }
        for (int j = 1; j < len; j++) cp = (cp << 6) | (s[i + j] & 63);
        starts.push_back((int)i);
        cps.push_back(cp);
        i += len;
    }
    starts.push_back((int)s.size());
    return cps;
}

struct Tokenizer {
    vector<string> pieces;                       // id -> bytes
    std::unordered_map<string, int> ids;
    std::unordered_map<uint64_t, std::pair<int, int>> merges;   // (a, b) -> (rank, merged id)
    vector<std::pair<string, int>> specials;
    int byte_id[256];
    bool digits = true, prefix_space = false, ignore_merges = false;
    std::unordered_map<string, vector<int>> cache;

    void load(const File& F) {
        auto vb = F.blob("tok.vocab");
        if (!vb.first) die("the model file has no tokenizer");
        const uint8_t* p = vb.first;
        uint32_t n;
        memcpy(&n, p, 4);
        const uint32_t* off = (const uint32_t*)(p + 4);
        const char* bytes = (const char*)(p + 4 + 4 * (n + 1));
        pieces.resize(n);
        for (uint32_t i = 0; i < n; i++) {
            pieces[i].assign(bytes + off[i], off[i + 1] - off[i]);
            ids.emplace(pieces[i], (int)i);
        }
        auto mb = F.blob("tok.merges");
        const int32_t* m = (const int32_t*)mb.first;
        for (size_t i = 0; i < mb.second / 12; i++)
            merges[((uint64_t)(uint32_t)m[3 * i] << 32) | (uint32_t)m[3 * i + 1]] = {(int)i, m[3 * i + 2]};
        auto sb = F.blob("tok.special");
        const int32_t* sp = (const int32_t*)sb.first;
        for (size_t i = 0; i < sb.second / 4; i++) specials.push_back({pieces[sp[i]], sp[i]});
        std::sort(specials.begin(), specials.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
        for (int b = 0; b < 256; b++) {
            auto it = ids.find(string(1, (char)b));
            byte_id[b] = it == ids.end() ? -1 : it->second;
        }
        digits = F.geti("tok_digits", 1);
        prefix_space = F.geti("tok_prefix_space", 0);
        ignore_merges = F.geti("tok_ignore_merges", 0);
    }

    void bpe(const string& w, vector<int>& out) {
        auto c = cache.find(w);
        if (c != cache.end()) { out.insert(out.end(), c->second.begin(), c->second.end()); return; }
        vector<int> sym;
        if (ignore_merges) {
            auto it = ids.find(w);
            if (it != ids.end()) sym.push_back(it->second);
        }
        if (sym.empty()) {
            for (unsigned char b : w) sym.push_back(byte_id[b]);
            for (;;) {
                int best = -1, rank = INT32_MAX, nid = -1;
                for (size_t i = 0; i + 1 < sym.size(); i++) {
                    auto it = merges.find(((uint64_t)(uint32_t)sym[i] << 32) | (uint32_t)sym[i + 1]);
                    if (it != merges.end() && it->second.first < rank) { rank = it->second.first; best = (int)i; nid = it->second.second; }
                }
                if (best < 0) break;
                // merge every occurrence of this pair, left to right (as the reference BPE does)
                int a = sym[best], b = sym[best + 1];
                vector<int> ns;
                for (size_t i = 0; i < sym.size(); i++) {
                    if (i + 1 < sym.size() && sym[i] == a && sym[i + 1] == b) { ns.push_back(nid); i++; }
                    else ns.push_back(sym[i]);
                }
                sym.swap(ns);
            }
        }
        if (cache.size() > 50000) cache.clear();
        cache[w] = sym;
        out.insert(out.end(), sym.begin(), sym.end());
    }

    // GPT-2's pattern over one stretch of text (no special pieces inside)
    void split_words(const string& s, vector<int>& out) {
        vector<int> st;
        vector<uint32_t> cp = utf8_decode(s, st);
        const int n = (int)cp.size();
        auto other = [&](uint32_t c) { return !is_S(c) && !is_L(c) && !is_N(c); };
        int i = 0;
        while (i < n) {
            int j = i;
            uint32_t c = cp[i];
            if (c == '\'' && i + 1 < n) {
                uint32_t a = cp[i + 1], b = i + 2 < n ? cp[i + 2] : 0;
                if (a == 's' || a == 't' || a == 'm' || a == 'd') j = i + 2;
                else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) j = i + 3;
            }
            if (j == i) {
                int k = (c == ' ' && i + 1 < n) ? i + 1 : i;
                uint32_t d = cp[k];
                if (is_L(d)) { j = k + 1; while (j < n && is_L(cp[j])) j++; }
                else if (is_N(d)) { j = k + 1; while (j < n && is_N(cp[j])) j++; }
                else if (other(d)) { j = k + 1; while (j < n && other(cp[j])) j++; }
                else if (k != i) { /* a space before a space: fall through to the whitespace rules */ }
            }
            if (j == i) {
                // whitespace: \s+(?!\S) then \s+
                int e = i;
                while (e < n && is_S(cp[e])) e++;
                if (e == n || e - i == 1) j = e;
                else j = e - 1;
                if (j == i) j = i + 1;
            }
            bpe(s.substr(st[i], st[j] - st[i]), out);
            i = j;
        }
    }

    vector<int> encode(const string& text) {
        vector<int> out;
        size_t i = 0, last = 0;
        auto plain = [&](const string& s) {
            if (s.empty()) return;
            string t = prefix_space && (s[0] != ' ') ? " " + s : s;
            if (!digits) { split_words(t, out); return; }
            size_t a = 0;
            for (size_t b = 0; b <= t.size(); b++) {     // each ASCII digit on its own
                if (b == t.size() || (t[b] >= '0' && t[b] <= '9')) {
                    if (b > a) split_words(t.substr(a, b - a), out);
                    if (b < t.size()) split_words(t.substr(b, 1), out);
                    a = b + 1;
                }
            }
        };
        while (i < text.size()) {
            bool hit = false;
            if (text[i] == '<') {
                for (auto& sp : specials)
                    if (text.compare(i, sp.first.size(), sp.first) == 0) {
                        plain(text.substr(last, i - last));
                        out.push_back(sp.second);
                        i += sp.first.size();
                        last = i;
                        hit = true;
                        break;
                    }
            }
            if (!hit) i++;
        }
        plain(text.substr(last));
        return out;
    }
};

// ====================================================================================== sampling
struct Sampler {
    std::mt19937_64 rng{(uint64_t)std::chrono::steady_clock::now().time_since_epoch().count()};
    float temp = 0.7f, top_p = 0.95f, rep = 1.1f;
    int top_k = 40;

    int pick(const float* logits, int V, const vector<int>& recent) {
        vector<float> l(logits, logits + V);
        if (rep != 1.f)
            for (int t : recent) if (t >= 0 && t < V) l[t] = l[t] > 0 ? l[t] / rep : l[t] * rep;
        if (temp <= 0) return (int)(std::max_element(l.begin(), l.end()) - l.begin());
        int k = top_k > 0 ? std::min(top_k, V) : V;
        vector<int> idx(V);
        for (int i = 0; i < V; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
        vector<double> p(k);
        double m = l[idx[0]], s = 0;
        for (int i = 0; i < k; i++) { p[i] = std::exp((l[idx[i]] - m) / temp); s += p[i]; }
        double cum = 0;
        int keep = k;
        for (int i = 0; i < k; i++) { cum += p[i] / s; if (cum >= top_p) { keep = i + 1; break; } }
        double tot = 0;
        for (int i = 0; i < keep; i++) tot += p[i];
        double r = std::uniform_real_distribution<double>(0, tot)(rng);
        for (int i = 0; i < keep; i++) { r -= p[i]; if (r <= 0) return idx[i]; }
        return idx[keep - 1];
    }
};

// ====================================================================================== the conversation
static string esc(const string& s) {
    string o;
    for (char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c == '\r') o += "\\r";
        else o += c;
    }
    return o;
}
static string unesc(const string& s) {
    string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char c = s[++i];
            o += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c;
        } else o += s[i];
    }
    return o;
}
static vector<string> split_tabs(const string& s) {
    vector<string> f;
    size_t a = 0;
    for (;;) {
        size_t b = s.find('\t', a);
        f.push_back(s.substr(a, b == string::npos ? string::npos : b - a));
        if (b == string::npos) break;
        a = b + 1;
    }
    return f;
}

static std::mutex OUT_MU;
static void say(const string& line) {
    std::lock_guard<std::mutex> lk(OUT_MU);
    fwrite(line.data(), 1, line.size(), stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

struct Chat {
    Model& M;
    Tokenizer& T;
    Sampler S;
    vector<int> history;           // every token read so far (to re-read the end when the context fills)
    vector<float> logits;
    bool need_sep = false;
    std::atomic<bool> stop{false};
    double t_read = 0;
    int n_read = 0;

    Chat(Model& m, Tokenizer& t) : M(m), T(t), logits(m.V) {}

    void reset() { M.reset(); history.clear(); need_sep = false; }

    void read(const vector<int>& toks) {
        double t0 = now();
        size_t i = 0;
        while (i < toks.size()) {
            if (M.pos + 1 > M.ctx) shrink();
            int n = (int)std::min<size_t>({(size_t)M.NB, toks.size() - i, (size_t)(M.ctx - M.pos)});
            M.forward(&toks[i], n, logits.data(), false);
            history.insert(history.end(), toks.begin() + i, toks.begin() + i + n);
            i += n;
        }
        t_read += now() - t0;
        n_read += (int)toks.size();
    }

    // The context is full: start again from the last third of the conversation (re-read once).
    void shrink() {
        vector<int> keep(history.end() - std::min<size_t>(history.size(), M.ctx / 3), history.end());
        M.reset();
        history.clear();
        say("note\tthe conversation filled the model's memory; it now remembers the most recent part");
        vector<int> again{M.eot};
        again.insert(again.end(), keep.begin(), keep.end());
        int i = 0;
        while (i < (int)again.size()) {
            int n = std::min<int>(M.NB, (int)again.size() - i);
            M.forward(&again[i], n, logits.data(), false);
            i += n;
        }
        history = again;
    }

    // mode: chat (User: / Assistant: turns, as the models saw them in training) or write (carry on the text)
    void run(const string& mode, const string& text, int max_new) {
        stop = false;
        t_read = 0;
        n_read = 0;
        vector<int> in;
        if (history.empty()) in.push_back(M.eot);           // "a new text starts here"
        auto add = [&](const string& s) { auto v = T.encode(s); in.insert(in.end(), v.begin(), v.end()); };
        if (mode == "chat") {
            add(string(need_sep ? "\n\n" : "") + "User:");
            add(" " + text);
            add("\n\nAssistant:");
        } else if (!text.empty()) add(text);
        read(in);
        need_sep = true;
        string out, pend;
        int made = 0;
        double t0 = now();
        while (made < max_new && !stop) {
            vector<int> recent(history.end() - std::min<size_t>(history.size(), 64), history.end());
            int t = S.pick(logits.data(), std::min(M.nvocab, M.V), recent);
            if (t == M.eot) break;
            string piece = T.pieces[t];
            string stripped = piece;
            stripped.erase(0, stripped.find_first_not_of(" \n\t"));
            if (!stripped.empty()) stripped.erase(stripped.find_last_not_of(" \n\t") + 1);
            if (mode == "chat" && (stripped == "User" || stripped == "User:") &&
                (out.empty() || out.find_first_not_of(" \n\t") == string::npos || out.back() == '\n')) {
                need_sep = false;                       // the model handed the turn back ("\n\n" already read)
                break;
            }
            out += piece;
            pend += piece;
            // send only whole UTF-8 characters
            size_t cut = pend.size();
            for (size_t k = pend.size(); k > 0 && k + 4 > pend.size(); k--) {
                unsigned char c = pend[k - 1];
                if ((c & 0xC0) == 0x80) continue;
                int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
                if (k - 1 + len > pend.size()) cut = k - 1;
                break;
            }
            if (cut > 0) { say("tok\t" + esc(pend.substr(0, cut))); pend.erase(0, cut); }
            made++;
            if (M.pos + 1 > M.ctx) shrink();
            vector<int> one{t};
            M.forward(one.data(), 1, logits.data(), false);
            history.push_back(t);
        }
        if (!pend.empty()) say("tok\t" + esc(pend));
        double dt = now() - t0;
        char buf[256];
        snprintf(buf, sizeof buf, "end\t%d\t%.2f\t%d\t%.2f\t%d\t%d", made, made > 0 && dt > 0 ? made / dt : 0.0, n_read,
                 t_read > 0 ? n_read / t_read : 0.0, M.pos, M.ctx);
        say(buf);
    }
};

// ====================================================================================== self-check
// The exporter stores texts with SmolLM2's own pieces for them, and PyTorch's scores after reading a few words.
static string selftest(Model& M, Tokenizer& T) {
    string rep;
    auto tt = M.F.blob("test.texts");
    auto ti = M.F.blob("test.ids");
    int ok = 0, tot = 0;
    if (tt.first && ti.first) {
        // texts: u32 count, then (u32 length, bytes)...; ids: per text (u32 n, ids...)
        const uint8_t* p = tt.first;
        const uint8_t* q = ti.first;
        uint32_t n;
        memcpy(&n, p, 4); p += 4;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t len, m;
            memcpy(&len, p, 4); p += 4;
            string s((const char*)p, len); p += len;
            memcpy(&m, q, 4); q += 4;
            vector<int> want((const int32_t*)q, (const int32_t*)q + m); q += 4 * m;
            tot++;
            if (T.encode(s) == want) ok++;
            else if (getenv("OAZY_DEBUG")) {
                auto got = T.encode(s);
                fprintf(stderr, "tokenizer mismatch on: %s\n want:", s.c_str());
                for (int x : want) fprintf(stderr, " %d", x);
                fprintf(stderr, "\n  got:");
                for (int x : got) fprintf(stderr, " %d", x);
                fprintf(stderr, "\n");
            }
        }
    }
    char buf[512];
    snprintf(buf, sizeof buf, "tokenizer %d/%d", ok, tot);
    rep = buf;
    const Mat* ref = M.F.opt("test.logits");
    auto rid = M.F.blob("test.prompt");
    if (ref && rid.first) {
        vector<int> toks((const int32_t*)rid.first, (const int32_t*)rid.first + rid.second / 4);
        M.reset();
        vector<float> lg(M.V);
        for (size_t i = 0; i < toks.size(); i += M.NB) {
            int n = std::min<int>(M.NB, (int)(toks.size() - i));
            M.forward(&toks[i], n, lg.data(), false);
        }
        M.reset();
        const float* r = ref->f;
        double num = 0, da = 0, db = 0, ma = 0, mb = 0, maxd = 0;
        for (int i = 0; i < M.V; i++) { ma += lg[i]; mb += r[i]; }
        ma /= M.V; mb /= M.V;
        for (int i = 0; i < M.V; i++) {
            num += (lg[i] - ma) * (r[i] - mb); da += (lg[i] - ma) * (lg[i] - ma); db += (r[i] - mb) * (r[i] - mb);
            maxd = std::max(maxd, (double)std::fabs(lg[i] - r[i]));
        }
        int a1 = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
        int b1 = (int)(std::max_element(r, r + M.V) - r);
        // top-10 overlap
        vector<int> ia(M.V), ib(M.V);
        for (int i = 0; i < M.V; i++) ia[i] = ib[i] = i;
        std::partial_sort(ia.begin(), ia.begin() + 10, ia.end(), [&](int x, int y) { return lg[x] > lg[y]; });
        std::partial_sort(ib.begin(), ib.begin() + 10, ib.end(), [&](int x, int y) { return r[x] > r[y]; });
        int ov = 0;
        for (int i = 0; i < 10; i++) for (int j = 0; j < 10; j++) if (ia[i] == ib[j]) ov++;
        snprintf(buf, sizeof buf, "; scores vs PyTorch: correlation %.4f, top word %s, top-10 shared %d/10, max diff %.3f",
                 num / std::sqrt(da * db + 1e-30), a1 == b1 ? "same" : "DIFFERENT", ov, maxd);
        rep += buf;
    }
    return rep;
}

// ====================================================================================== main
static void usage() {
    fprintf(stderr,
            "oazy engine\n"
            "  --model FILE [--threads N] [--ctx N] serve          talk to the app over stdin/stdout\n"
            "  --model FILE selftest                               tokenizer and scores against PyTorch\n"
            "  --model FILE encode TEXT                            token ids\n"
            "  --model FILE logits [--chunk N] ID ID ...           scores after every word (binary floats to stdout)\n"
            "  --model FILE bench [N]                              words per second\n");
    exit(2);
}

int main(int argc, char** argv) {
    string model, cmd;
    int threads = 0, ctx = 0, chunk = 32;
    vector<string> rest;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--threads" && i + 1 < argc) threads = atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (a == "--chunk" && i + 1 < argc) chunk = atoi(argv[++i]);
        else if (cmd.empty()) cmd = a;
        else rest.push_back(a);
    }
    if (model.empty() || cmd.empty()) usage();
    pick_kernels();
    if (threads <= 0) {
        int hc = (int)std::thread::hardware_concurrency();
        threads = hc >= 8 ? 4 : std::max(1, hc / 2);
    }
    if (cmd == "serve") pin_to_fast_cores(threads);
    POOL.start(threads);

    Model M;
    double t0 = now();
    M.NB = std::max(1, std::min(chunk, 64));
    M.load(model.c_str(), ctx);
    Tokenizer T;
    T.load(M.F);

    if (cmd == "encode") {
        string s;
        for (size_t i = 0; i < rest.size(); i++) s += (i ? " " : "") + rest[i];
        for (int t : T.encode(s)) printf("%d ", t);
        printf("\n");
        return 0;
    }
    if (cmd == "selftest") {
        printf("%s\n", selftest(M, T).c_str());
        return 0;
    }
    if (cmd == "logits") {
        vector<int> toks;
        for (auto& r : rest) toks.push_back(atoi(r.c_str()));
        vector<float> out((size_t)M.NB * M.V);
        for (size_t i = 0; i < toks.size(); i += M.NB) {
            int n = std::min<int>(M.NB, (int)(toks.size() - i));
            M.forward(&toks[i], n, out.data(), true);
            fwrite(out.data(), sizeof(float), (size_t)n * M.V, stdout);
        }
        return 0;
    }
    if (cmd == "bench") {
        int n = rest.empty() ? 64 : atoi(rest[0].c_str());
        vector<int> p(128);
        for (int i = 0; i < 128; i++) p[i] = 100 + i;
        vector<float> lg(M.V);
        double a = now();
        for (int i = 0; i < 128; i += M.NB) M.forward(&p[i], std::min(M.NB, 128 - i), lg.data(), false);
        double b = now();
        for (int i = 0; i < n; i++) { int t = 200 + i; M.forward(&t, 1, lg.data(), false); }
        double c = now();
        printf("%s, %s, %d threads: reading %.1f words/s, writing %.2f words/s\n", M.name.c_str(), KERNEL, POOL.n, 128 / (b - a), n / (c - b));
        return 0;
    }
    if (cmd != "serve") usage();

    Chat C(M, T);
    {
        char buf[1024];
        snprintf(buf, sizeof buf,
                 "ready\tname=%s\tkind=%s\tquant=%s\tparams=%s\tactive=%s\tlayers=%d\tctx=%d\ttokens=%s\tthreads=%d\tkernel=%s\tload=%.2f\tsize=%zu\tlabel=%s\taccent=%s",
                 M.name.c_str(), M.kind.c_str(), M.quant.c_str(), M.F.get("params").c_str(), M.F.get("active").c_str(), M.L,
                 M.ctx, M.F.get("tokens_trained").c_str(), POOL.n, KERNEL, now() - t0, M.F.size, M.F.get("label").c_str(),
                 M.F.get("accent").c_str());
        say(buf);
    }
    // commands arrive on stdin; "stop" must reach a running answer, so a reader thread hands the rest over
    std::mutex qmu;
    std::condition_variable qcv;
    std::vector<string> queue;
    bool eof = false;
    std::thread reader([&] {
        char* line = nullptr;
        size_t cap = 0;
        ssize_t len;
        while ((len = getline(&line, &cap, stdin)) >= 0) {
            string s(line, len);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (s == "stop") { C.stop = true; continue; }
            std::lock_guard<std::mutex> lk(qmu);
            queue.push_back(s);
            qcv.notify_one();
        }
        std::lock_guard<std::mutex> lk(qmu);
        eof = true;
        qcv.notify_one();
    });
    reader.detach();
    for (;;) {
        string s;
        {
            std::unique_lock<std::mutex> lk(qmu);
            qcv.wait(lk, [&] { return !queue.empty() || eof; });
            if (queue.empty()) break;
            s = queue.front();
            queue.erase(queue.begin());
        }
        auto f = split_tabs(s);
        if (f[0] == "chat" || f[0] == "write") {
            // chat|write <max new> <temperature> <top k> <top p> <repetition penalty> <text>
            if (f.size() < 7) { say("error\tbad command"); continue; }
            C.S.temp = (float)atof(f[2].c_str());
            C.S.top_k = atoi(f[3].c_str());
            C.S.top_p = (float)atof(f[4].c_str());
            C.S.rep = (float)atof(f[5].c_str());
            C.run(f[0], unesc(f[6]), atoi(f[1].c_str()));
        } else if (f[0] == "reset") {
            C.reset();
            say("ok");
        } else if (f[0] == "selftest") {
            C.reset();
            say("selftest\t" + selftest(M, T));
        } else if (f[0] == "quit") {
            break;
        } else {
            say("error\tunknown command " + esc(f[0]));
        }
    }
    return 0;
}
