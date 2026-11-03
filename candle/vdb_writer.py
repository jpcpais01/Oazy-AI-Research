"""Minimal, dependency-free OpenVDB (.vdb) writer: numpy dense arrays -> sparse grids.

Writes file format version 224 with the standard Tree_<type>_5_4_3 layout,
ZIP-compressed node buffers (no active-mask compression), a uniform
scale+translate transform, and float or vec3s grids.  Only voxels inside 8^3
leaves that contain non-zero data are stored, so sparse flame/smoke data stays
small.  Verified against the reference OpenVDB library and Blender/Cycles.

Why not pyopenvdb?  Neither the pip `bpy` module nor PyPI ship OpenVDB Python
bindings, and keeping this stage pure numpy makes the pipeline portable.
"""
from __future__ import annotations

import io
import struct
import uuid
import zlib

import numpy as np

FILE_VERSION = 224
LIB_MAJOR, LIB_MINOR = 11, 0
COMPRESS_ZIP = 0x1
NO_MASK_AND_ALL_VALS = 6

LEAF_LOG2, INT1_LOG2, INT2_LOG2 = 3, 4, 5
LEAF_DIM = 1 << LEAF_LOG2                      # 8 voxels
INT1_SPAN = LEAF_DIM << INT1_LOG2              # 128 voxels
INT2_SPAN = INT1_SPAN << INT2_LOG2             # 4096 voxels


def _wstr(buf, s: str):
    b = s.encode("utf-8")
    buf.write(struct.pack("<I", len(b)))
    buf.write(b)


def _zip(buf, arr: np.ndarray):
    raw = np.ascontiguousarray(arr).tobytes()
    z = zlib.compress(raw, 4)
    if len(z) < len(raw):
        buf.write(struct.pack("<q", len(z)))
        buf.write(z)
    else:
        buf.write(struct.pack("<q", -len(raw)))
        buf.write(raw)


def _mask_bytes(bits: np.ndarray) -> bytes:
    """bits: flat bool array of length N (N multiple of 64), bit n -> word n//64, bit n%64."""
    b = np.packbits(bits.astype(np.uint8), bitorder="little")
    return b.tobytes()


def _write_values(buf, values: np.ndarray):
    buf.write(struct.pack("<b", NO_MASK_AND_ALL_VALS))
    _zip(buf, values)


class _Grid:
    def __init__(self, name, data, threshold):
        data = np.asarray(data)
        self.name = name
        if data.ndim == 4:
            assert data.shape[-1] == 3
            self.vec = True
            self.data = data.astype(np.float32)
            self.mag = np.abs(self.data).max(-1)
        else:
            self.vec = False
            self.data = data.astype(np.float32)
            self.mag = np.abs(self.data)
        self.threshold = threshold

    @property
    def type_name(self):
        return "Tree_vec3s_5_4_3" if self.vec else "Tree_float_5_4_3"

    def _zero(self):
        return np.zeros(3, np.float32) if self.vec else np.float32(0)


def _build_tree(g: _Grid):
    """Returns dict int1_origin -> list of (leaf_origin, leaf_values, leaf_mask)."""
    nx, ny, nz = g.mag.shape
    px, py, pz = (-(-n // LEAF_DIM) * LEAF_DIM for n in (nx, ny, nz))
    if max(px, py, pz) > INT2_SPAN:
        raise ValueError("grid larger than one 4096^3 internal node is not supported")
    mag = np.zeros((px, py, pz), np.float32)
    mag[:nx, :ny, :nz] = g.mag
    L = mag.reshape(px // 8, 8, py // 8, 8, pz // 8, 8).transpose(0, 2, 4, 1, 3, 5)
    occupied = (L > g.threshold).any(axis=(3, 4, 5))
    if g.vec:
        dat = np.zeros((px, py, pz, 3), np.float32)
        dat[:nx, :ny, :nz] = g.data
        D = dat.reshape(px // 8, 8, py // 8, 8, pz // 8, 8, 3).transpose(0, 2, 4, 1, 3, 5, 6)
    else:
        dat = np.zeros((px, py, pz), np.float32)
        dat[:nx, :ny, :nz] = g.data
        D = dat.reshape(px // 8, 8, py // 8, 8, pz // 8, 8).transpose(0, 2, 4, 1, 3, 5)
    nodes = {}
    for li, lj, lk in zip(*np.nonzero(occupied)):
        ox, oy, oz = int(li) * 8, int(lj) * 8, int(lk) * 8
        key = (ox // INT1_SPAN * INT1_SPAN, oy // INT1_SPAN * INT1_SPAN, oz // INT1_SPAN * INT1_SPAN)
        vals = D[li, lj, lk]
        active = L[li, lj, lk] > g.threshold
        if g.vec:
            vals = np.where(active[..., None], vals, 0.0)
        else:
            vals = np.where(active, vals, 0.0)
        nodes.setdefault(key, []).append(((ox, oy, oz), vals.astype(np.float32), active))
    return nodes


def _int1_index(origin_leaf, origin_int1):
    x = (origin_leaf[0] - origin_int1[0]) >> LEAF_LOG2
    y = (origin_leaf[1] - origin_int1[1]) >> LEAF_LOG2
    z = (origin_leaf[2] - origin_int1[2]) >> LEAF_LOG2
    return (x << (2 * INT1_LOG2)) + (y << INT1_LOG2) + z


def _int2_index(origin_int1):
    x = origin_int1[0] // INT1_SPAN
    y = origin_int1[1] // INT1_SPAN
    z = origin_int1[2] // INT1_SPAN
    return (x << (2 * INT2_LOG2)) + (y << INT2_LOG2) + z


def _write_grid(buf: io.BytesIO, g: _Grid, voxel_size: float, translation):
    _wstr(buf, g.name)
    _wstr(buf, g.type_name)
    _wstr(buf, "")                              # instance parent
    offset_pos = buf.tell()
    buf.write(struct.pack("<qqq", 0, 0, 0))     # grid / block / end positions (patched)
    grid_pos = buf.tell()
    buf.write(struct.pack("<I", COMPRESS_ZIP))  # per-grid compression flags
    # grid metadata
    meta = [("class", "string", b"fog volume"), ("name", "string", g.name.encode())]
    buf.write(struct.pack("<i", len(meta)))
    for k, t, v in meta:
        _wstr(buf, k)
        _wstr(buf, t)
        buf.write(struct.pack("<I", len(v)))
        buf.write(v)
    # transform
    _wstr(buf, "UniformScaleTranslateMap")
    # ScaleTranslateMap::write: translation, scale, voxelSize, 1/scale, 1/scale^2, 1/(2 scale)
    vs = float(voxel_size)
    buf.write(struct.pack("<3d", *[float(t) for t in translation]))
    for val in (vs, vs, 1.0 / vs, 1.0 / (vs * vs), 0.5 / vs):
        buf.write(struct.pack("<3d", val, val, val))

    # ---- topology ----------------------------------------------------------
    nodes = _build_tree(g)
    zero = g._zero()
    buf.write(struct.pack("<i", 1))             # buffer count
    if g.vec:
        buf.write(struct.pack("<3f", 0.0, 0.0, 0.0))
    else:
        buf.write(struct.pack("<f", 0.0))       # background
    buf.write(struct.pack("<I", 0))             # root tiles
    leaves_in_order = []
    if not nodes:
        buf.write(struct.pack("<I", 0))         # no children
    else:
        buf.write(struct.pack("<I", 1))         # one InternalNode<5> at the origin
        buf.write(struct.pack("<3i", 0, 0, 0))
        # InternalNode<5>
        n2 = 1 << (3 * INT2_LOG2)
        child2 = np.zeros(n2, bool)
        order1 = sorted(nodes.keys(), key=_int2_index)
        for key in order1:
            child2[_int2_index(key)] = True
        buf.write(_mask_bytes(child2))
        buf.write(_mask_bytes(np.zeros(n2, bool)))
        vshape = (n2, 3) if g.vec else (n2,)
        _write_values(buf, np.zeros(vshape, np.float32))
        n1 = 1 << (3 * INT1_LOG2)
        for key in order1:
            leaves = sorted(nodes[key], key=lambda l: _int1_index(l[0], key))
            child1 = np.zeros(n1, bool)
            for l in leaves:
                child1[_int1_index(l[0], key)] = True
            buf.write(_mask_bytes(child1))
            buf.write(_mask_bytes(np.zeros(n1, bool)))
            vshape = (n1, 3) if g.vec else (n1,)
            _write_values(buf, np.zeros(vshape, np.float32))
            for l in leaves:
                buf.write(_mask_bytes(l[2].reshape(-1)))     # leaf topology = value mask
                leaves_in_order.append(l)
    block_pos = buf.tell()
    # ---- buffers -------------------------------------------------------------
    for origin, vals, active in leaves_in_order:
        buf.write(_mask_bytes(active.reshape(-1)))
        _write_values(buf, vals.reshape(-1, 3) if g.vec else vals.reshape(-1))
    end_pos = buf.tell()
    here = buf.tell()
    buf.seek(offset_pos)
    buf.write(struct.pack("<qqq", grid_pos, block_pos, end_pos))
    buf.seek(here)
    del zero


def write_vdb(path, grids: dict, voxel_size: float, translation=(0.0, 0.0, 0.0), threshold=1e-6):
    """grids: name -> ndarray (nx,ny,nz) float or (nx,ny,nz,3) vector, indexed [x,y,z].

    Voxel (i,j,k) is centred at translation + voxel_size*(i,j,k) in world space.
    """
    buf = io.BytesIO()
    buf.write(struct.pack("<q", 0x56444220))
    buf.write(struct.pack("<III", FILE_VERSION, LIB_MAJOR, LIB_MINOR))
    buf.write(struct.pack("<b", 1))             # has grid offsets
    buf.write(str(uuid.uuid4()).encode("ascii"))
    buf.write(struct.pack("<i", 0))             # file-level metadata count
    buf.write(struct.pack("<i", len(grids)))
    for name, data in grids.items():
        _write_grid(buf, _Grid(name, data, threshold), voxel_size, translation)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
    import os
    os.replace(tmp, path)
