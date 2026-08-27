"""Turn a trellis .ply (the pre-remesh mesh every run writes) into a minimal
post-replay dump, so the geometry half of the pipeline can be replayed on any
asset without a GPU.

The dump carries no PBR volume (Mv=0), so the result is geometry only - run
post-replay with --no-bake. That is enough to exercise weld, fill_small_holes,
remesh_dc and strip_interior, which is what we want to test for consistency
across assets.

Usage: python ply_to_dump.py <in.ply> <out.dump> [res]
"""
import struct
import sys


def read_ply(path):
    with open(path, "rb") as fh:
        magic = fh.readline().strip()
        if magic != b"ply":
            raise SystemExit("%s: not a ply" % path)
        fmt = None
        nvert = nface = 0
        vprops = []
        in_vert = in_face = False
        face_count_type = face_index_type = None
        while True:
            line = fh.readline()
            if not line:
                raise SystemExit("%s: no end_header" % path)
            t = line.split()
            if not t:
                continue
            if t[0] == b"format":
                fmt = t[1]
            elif t[0] == b"element":
                if t[1] == b"vertex":
                    nvert = int(t[2]); in_vert, in_face = True, False
                elif t[1] == b"face":
                    nface = int(t[2]); in_vert, in_face = False, True
                else:
                    in_vert = in_face = False
            elif t[0] == b"property":
                if in_vert and t[1] != b"list":
                    vprops.append((t[1].decode(), t[2].decode()))
                elif in_face and t[1] == b"list":
                    face_count_type, face_index_type = t[2].decode(), t[3].decode()
            elif t[0] == b"end_header":
                break
        if fmt != b"binary_little_endian":
            raise SystemExit("%s: only binary_little_endian is handled (got %s)" % (path, fmt))

        SZ = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1,
              "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
              "int": 4, "uint": 4, "int32": 4, "uint32": 4, "float": 4, "float32": 4,
              "double": 8, "float64": 8}
        CODE = {"char": "b", "uchar": "B", "int8": "b", "uint8": "B",
                "short": "h", "ushort": "H", "int16": "h", "uint16": "H",
                "int": "i", "uint": "I", "int32": "i", "uint32": "I",
                "float": "f", "float32": "f", "double": "d", "float64": "d"}

        stride = sum(SZ[t] for t, _ in vprops)
        names = [n for _, n in vprops]
        off = {}
        acc = 0
        for t, n in vprops:
            off[n] = (acc, CODE[t])
            acc += SZ[t]
        for k in ("x", "y", "z"):
            if k not in off:
                raise SystemExit("%s: vertex property %s missing" % (path, k))

        # numpy only - these meshes run to millions of faces and a per-element
        # Python loop takes minutes. Run under Blender's bundled interpreter:
        #   "...\Blender 5.1\5.1\python\bin\python.exe" ply_to_dump.py ...
        import numpy as np

        raw = fh.read(stride * nvert)
        if len(raw) != stride * nvert:
            raise SystemExit("%s: truncated vertex block" % path)
        NPT = {"char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
               "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
               "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
               "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}
        vdt = np.dtype([(n, NPT[t]) for t, n in vprops])
        if vdt.itemsize != stride:
            raise SystemExit("%s: vertex stride mismatch" % path)
        va = np.frombuffer(raw, dtype=vdt, count=nvert)
        xyz = np.empty((nvert, 3), dtype="<f4")
        for k, name in enumerate(("x", "y", "z")):
            xyz[:, k] = va[name].astype("<f4")

        fraw = fh.read()
        cs, isz = SZ[face_count_type], SZ[face_index_type]
        tri_rec = cs + 3 * isz
        fdt = np.dtype([("n", NPT[face_count_type]), ("i", NPT[face_index_type], 3)])
        if len(fraw) == nface * tri_rec and fdt.itemsize == tri_rec:
            fa = np.frombuffer(fraw, dtype=fdt, count=nface)
            if not np.all(fa["n"] == 3):
                raise SystemExit("%s: mixed polygon sizes, not handled" % path)
            tris = fa["i"].astype("<i4")
        else:
            raise SystemExit("%s: face block is not all triangles (%d bytes for %d faces)"
                             % (path, len(fraw), nface))
        return nvert, xyz.tobytes(), len(tris), tris.tobytes()


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    res = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    nv, verts, nf, faces = read_ply(src)
    print("%s: V=%d F=%d -> %s (res=%d, no PBR volume)" % (src, nv, nf, dst, res))
    with open(dst, "wb") as fh:
        fh.write(struct.pack("<4i", nv, nf, 0, res))
        fh.write(verts)
        fh.write(faces)


if __name__ == "__main__":
    main()
