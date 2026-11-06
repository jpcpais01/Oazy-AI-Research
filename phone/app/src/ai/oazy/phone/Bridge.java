package ai.oazy.phone;

import android.app.ActivityManager;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.provider.OpenableColumns;
import android.webkit.JavascriptInterface;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Arrays;

/**
 * What the page can ask of the phone (window.OazyNative in assets/app.js), and the engine process.
 *
 * The engine (lib/arm64-v8a/liboazy.so, built from phone/engine/oazy.cpp) is a small native program, started once per
 * model, that maps the model file into memory and answers over stdin / stdout one line at a time:
 *   in   chat|write \t max_new \t temperature \t top_k \t top_p \t repetition \t text      reset   selftest   stop
 *   out  ready \t key=value ...    tok \t text    end \t words \t speed \t read \t read speed \t used \t ctx
 *        note \t text    selftest \t report    error \t text
 * Every line becomes an event for the page: {"type": "tok", "text": "..."} and so on.
 */
public class Bridge {
    static final Charset UTF8 = Charset.forName("UTF-8");
    final MainActivity act;
    Process proc;
    Writer toEngine;
    String loaded;                      // the model file the engine has open
    volatile boolean cancelDownload, cancelImport;

    Bridge(MainActivity a) { act = a; }

    File modelDir() {
        File d = act.getExternalFilesDir("models");
        if (d == null) d = new File(act.getFilesDir(), "models");
        d.mkdirs();
        return d;
    }

    void send(String type, Object... kv) {
        try {
            JSONObject o = new JSONObject();
            o.put("type", type);
            for (int i = 0; i + 1 < kv.length; i += 2) o.put((String) kv[i], kv[i + 1]);
            act.event(o);
        } catch (Exception e) { /* cannot happen */ }
    }

    // ------------------------------------------------------------------------------------------ the phone
    @JavascriptInterface
    public String device() {
        try {
            ActivityManager am = (ActivityManager) act.getSystemService(Context.ACTIVITY_SERVICE);
            ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
            am.getMemoryInfo(mi);
            JSONObject o = new JSONObject();
            o.put("ram", mi.totalMem);
            o.put("free", mi.availMem);
            o.put("cores", Runtime.getRuntime().availableProcessors());
            o.put("fast", fastCores());
            o.put("model", Build.MANUFACTURER + " " + Build.MODEL);
            o.put("android", Build.VERSION.RELEASE);
            o.put("soc", Build.HARDWARE);
            o.put("dir", modelDir().getAbsolutePath());
            return o.toString();
        } catch (Exception e) { return "{}"; }
    }

    /** The big cores: those at 70% or more of the fastest core's top speed (phones mix fast and slow cores). */
    static int fastCores() {
        long[] f = new long[64];
        int n = 0;
        long top = 0;
        for (int c = 0; c < 64; c++) {
            File p = new File("/sys/devices/system/cpu/cpu" + c + "/cpufreq/cpuinfo_max_freq");
            if (!p.exists()) continue;
            try {
                BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(p)));
                f[n] = Long.parseLong(r.readLine().trim());
                r.close();
                top = Math.max(top, f[n]);
                n++;
            } catch (Exception e) { /* hidden on some phones */ }
        }
        int big = 0;
        for (int i = 0; i < n; i++) if (f[i] >= top * 0.7) big++;
        int cores = Runtime.getRuntime().availableProcessors();
        if (big == 0) big = Math.max(1, cores / 2);
        return Math.max(2, Math.min(big, 6));
    }

    // ------------------------------------------------------------------------------------------ the model files
    /** The header of a model file: its settings as a JSON object (null if it is not one). */
    static JSONObject header(File f) {
        try {
            FileInputStream in = new FileInputStream(f);
            byte[] head = new byte[16];
            if (in.read(head) != 16 || !new String(head, 0, 8, UTF8).equals("OAZYPHN1")) { in.close(); return null; }
            long len = 0;
            for (int i = 15; i >= 8; i--) len = (len << 8) | (head[i] & 0xff);
            byte[] h = new byte[(int) Math.min(len, 1 << 22)];
            int got = 0;
            while (got < h.length) { int r = in.read(h, got, h.length - got); if (r < 0) break; got += r; }
            in.close();
            JSONObject o = new JSONObject();
            for (String line : new String(h, 0, got, UTF8).split("\n")) {
                if (line.startsWith("tensor ") || line.isEmpty()) continue;
                int sp = line.indexOf(' ');
                if (sp > 0) o.put(line.substring(0, sp), line.substring(sp + 1));
            }
            return o;
        } catch (Exception e) { return null; }
    }

    @JavascriptInterface
    public String models() {
        JSONArray a = new JSONArray();
        File[] fs = modelDir().listFiles();
        if (fs != null) {
            Arrays.sort(fs);
            for (File f : fs) {
                if (!f.getName().endsWith(".oazy")) continue;
                JSONObject h = header(f);
                if (h == null) continue;
                try {
                    h.put("file", f.getName());
                    h.put("bytes", f.length());
                    h.put("loaded", f.getName().equals(loaded));
                    a.put(h);
                } catch (Exception e) { /* skip */ }
            }
        }
        return a.toString();
    }

    @JavascriptInterface
    public boolean deleteModel(String name) {
        File f = new File(modelDir(), new File(name).getName());
        if (f.getName().equals(loaded)) stopEngine();
        return f.delete();
    }

    @JavascriptInterface
    public void pickModel() {
        act.runOnUiThread(new Runnable() { public void run() { act.pickModel(); } });
    }

    @JavascriptInterface
    public void cancelTransfer() { cancelDownload = true; cancelImport = true; }

    /** Copy a model the user picked (Downloads, a USB stick, ...) into the app's model folder. */
    void importFrom(final Uri uri) {
        cancelImport = false;
        new Thread(new Runnable() {
            public void run() {
                String name = "model.oazy";
                long total = -1;
                try {
                    Cursor c = act.getContentResolver().query(uri, null, null, null, null);
                    if (c != null) {
                        if (c.moveToFirst()) {
                            int ni = c.getColumnIndex(OpenableColumns.DISPLAY_NAME), si = c.getColumnIndex(OpenableColumns.SIZE);
                            if (ni >= 0) name = c.getString(ni);
                            if (si >= 0) total = c.getLong(si);
                        }
                        c.close();
                    }
                    name = new File(name).getName();
                    if (!name.endsWith(".oazy")) name = name.replaceAll("\\.[A-Za-z0-9]+$", "") + ".oazy";
                    File part = new File(modelDir(), name + ".part");
                    InputStream in = act.getContentResolver().openInputStream(uri);
                    byte[] head = new byte[8];
                    int hn = in.read(head);
                    if (hn != 8 || !new String(head, UTF8).equals("OAZYPHN1")) {
                        in.close();
                        send("transfer", "state", "error", "what", "import", "message", "This file is not an Oazy phone model (.oazy). Make one with python3 -m phone.export on the pod.");
                        return;
                    }
                    OutputStream out = new FileOutputStream(part);
                    out.write(head);
                    copy(in, out, 8, total, "import", name);
                    in.close();
                    out.close();
                    if (cancelImport) { part.delete(); send("transfer", "state", "cancelled", "what", "import"); return; }
                    File dst = new File(modelDir(), name);
                    dst.delete();
                    part.renameTo(dst);
                    send("transfer", "state", "done", "what", "import", "file", name);
                } catch (Exception e) {
                    send("transfer", "state", "error", "what", "import", "message", String.valueOf(e.getMessage()));
                }
            }
        }).start();
    }

    void copy(InputStream in, OutputStream out, long done, long total, String what, String name) throws Exception {
        byte[] buf = new byte[1 << 20];
        long last = 0;
        int r;
        while ((r = in.read(buf)) > 0) {
            if ((what.equals("import") && cancelImport) || (what.equals("download") && cancelDownload)) return;
            out.write(buf, 0, r);
            done += r;
            long now = System.currentTimeMillis();
            if (now - last > 250) {
                last = now;
                send("transfer", "state", "progress", "what", what, "file", name, "done", done, "total", total);
            }
        }
    }

    /** Download a model from a link (the pod's share page: python3 -m phone.serve), resuming a cut download. */
    @JavascriptInterface
    public void download(final String link) {
        cancelDownload = false;
        new Thread(new Runnable() {
            public void run() {
                try {
                    String name = Uri.parse(link).getLastPathSegment();
                    if (name == null || !name.endsWith(".oazy")) name = "model.oazy";
                    name = new File(name).getName();
                    File part = new File(modelDir(), name + ".part");
                    long have = part.exists() ? part.length() : 0;
                    URL url = new URL(link);
                    HttpURLConnection c = null;
                    for (int hop = 0; hop < 5; hop++) {           // follow redirects, also http <-> https
                        c = (HttpURLConnection) url.openConnection();
                        c.setConnectTimeout(20000);
                        c.setReadTimeout(60000);
                        c.setInstanceFollowRedirects(false);
                        if (have > 0) c.setRequestProperty("Range", "bytes=" + have + "-");
                        int code = c.getResponseCode();
                        if (code / 100 == 3 && c.getHeaderField("Location") != null) {
                            url = new URL(url, c.getHeaderField("Location"));
                            c.disconnect();
                            continue;
                        }
                        break;
                    }
                    int code = c.getResponseCode();
                    if (code != 200 && code != 206) {
                        send("transfer", "state", "error", "what", "download", "message", "The link answered " + code + " " + c.getResponseMessage());
                        return;
                    }
                    if (code == 200) have = 0;                     // the server started from the beginning
                    long len = -1;
                    try { len = Long.parseLong(c.getHeaderField("Content-Length")); } catch (Exception e) { /* unknown */ }
                    long total = len >= 0 ? have + len : -1;
                    InputStream in = c.getInputStream();
                    OutputStream out = new FileOutputStream(part, have > 0);
                    copy(in, out, have, total, "download", name);
                    in.close();
                    out.close();
                    c.disconnect();
                    if (cancelDownload) { send("transfer", "state", "cancelled", "what", "download"); return; }
                    if (total > 0 && part.length() != total) {
                        send("transfer", "state", "error", "what", "download", "message", "The download stopped early; tap Download again to carry on from where it stopped.");
                        return;
                    }
                    if (header(part) == null) {
                        part.delete();
                        send("transfer", "state", "error", "what", "download", "message", "That link did not give an Oazy phone model (.oazy).");
                        return;
                    }
                    File dst = new File(modelDir(), name);
                    dst.delete();
                    part.renameTo(dst);
                    send("transfer", "state", "done", "what", "download", "file", name);
                } catch (Exception e) {
                    send("transfer", "state", "error", "what", "download", "message", e.getClass().getSimpleName() + ": " + e.getMessage() + " (tap Download again to resume)");
                }
            }
        }).start();
    }

    // ------------------------------------------------------------------------------------------ the engine
    @JavascriptInterface
    public void load(final String name, final int threads, final int ctx) {
        stopEngine();
        final File f = new File(modelDir(), new File(name).getName());
        new Thread(new Runnable() {
            public void run() {
                try {
                    String exe = act.getApplicationInfo().nativeLibraryDir + "/liboazy.so";
                    ProcessBuilder pb = new ProcessBuilder(exe, "--model", f.getAbsolutePath(), "--threads",
                            String.valueOf(threads > 0 ? threads : fastCores()), "--ctx", String.valueOf(Math.max(0, ctx)), "serve");
                    final Process p = pb.start();
                    synchronized (Bridge.this) {
                        proc = p;
                        toEngine = new OutputStreamWriter(p.getOutputStream(), UTF8);
                        loaded = f.getName();
                    }
                    new Thread(new Runnable() {     // the engine's few words on stderr, for the log
                        public void run() {
                            try {
                                BufferedReader r = new BufferedReader(new InputStreamReader(p.getErrorStream(), UTF8));
                                String l;
                                while ((l = r.readLine()) != null) android.util.Log.w("oazy-engine", l);
                            } catch (Exception e) { /* closed */ }
                        }
                    }).start();
                    BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), UTF8));
                    String line;
                    while ((line = r.readLine()) != null) onLine(line);
                    int code = p.waitFor();
                    synchronized (Bridge.this) { if (proc != p) return; proc = null; loaded = null; }
                    send("exit", "code", code);
                } catch (Exception e) {
                    send("error", "text", "The engine could not start: " + e.getMessage());
                }
            }
        }).start();
    }

    void onLine(String line) {
        String[] f = line.split("\t", -1);
        try {
            JSONObject o = new JSONObject();
            o.put("type", f[0]);
            if (f[0].equals("ready")) {
                for (int i = 1; i < f.length; i++) {
                    int eq = f[i].indexOf('=');
                    if (eq > 0) o.put(f[i].substring(0, eq), f[i].substring(eq + 1));
                }
                o.put("file", loaded);
            } else if (f[0].equals("tok") || f[0].equals("note") || f[0].equals("error") || f[0].equals("selftest")) {
                o.put("text", f.length > 1 ? unesc(f[1]) : "");
            } else if (f[0].equals("end") && f.length >= 7) {
                o.put("words", Integer.parseInt(f[1]));
                o.put("speed", Double.parseDouble(f[2]));
                o.put("read", Integer.parseInt(f[3]));
                o.put("readSpeed", Double.parseDouble(f[4]));
                o.put("used", Integer.parseInt(f[5]));
                o.put("ctx", Integer.parseInt(f[6]));
            }
            act.event(o);
        } catch (Exception e) { /* a line we do not know */ }
    }

    static String unesc(String s) {
        StringBuilder b = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char d = s.charAt(++i);
                b.append(d == 'n' ? '\n' : d == 't' ? '\t' : d == 'r' ? '\r' : d);
            } else b.append(c);
        }
        return b.toString();
    }

    static String esc(String s) {
        return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r");
    }

    synchronized boolean write(String line) {
        if (toEngine == null) return false;
        try {
            toEngine.write(line + "\n");
            toEngine.flush();
            return true;
        } catch (Exception e) { return false; }
    }

    @JavascriptInterface
    public boolean ask(String mode, String text, int maxNew, double temp, int topK, double topP, double rep) {
        return write(("write".equals(mode) ? "write" : "chat") + "\t" + maxNew + "\t" + temp + "\t" + topK + "\t" + topP + "\t" + rep + "\t" + esc(text));
    }

    @JavascriptInterface
    public void stop() { write("stop"); }

    @JavascriptInterface
    public void reset() { write("reset"); }

    @JavascriptInterface
    public void selftest() { write("selftest"); }

    @JavascriptInterface
    public void unload() { stopEngine(); }

    synchronized void stopEngine() {
        if (proc != null) {
            Process p = proc;
            proc = null;
            loaded = null;
            try { toEngine.write("quit\n"); toEngine.flush(); } catch (Exception e) { /* gone */ }
            p.destroy();
        }
        toEngine = null;
    }

    void shutdown() { cancelDownload = true; cancelImport = true; stopEngine(); }
}
