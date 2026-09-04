#include "master_cache.h"
#include "atomic_file.h"
#include <cstdio>

namespace trellis {

namespace {

// Deliberately negative as a signed 32-bit value. The legacy layout opens with
// V, a vertex count, which is always positive -- so one read distinguishes the
// two formats with no ambiguity and no guessing at plausible magnitudes.
constexpr uint32_t MASTER_MAGIC   = 0x9E3779B9u;
constexpr uint32_t MASTER_VERSION = 1u;

template <typename T> bool rd(FILE* f, T& v) { return fread(&v, sizeof(T), 1, f) == 1; }
template <typename T> bool wr(FILE* f, const T& v) { return fwrite(&v, sizeof(T), 1, f) == 1; }

bool read_body(FILE* f, Master& m, int V, int F, int Mv) {
    m.verts.resize((size_t)V * 3);
    m.faces.resize((size_t)F * 3);
    m.coords.resize((size_t)Mv);
    m.pbr6.resize((size_t)Mv * 6);
    if (fread(m.verts.data(), 4, m.verts.size(), f) != m.verts.size()) return false;
    if (fread(m.faces.data(), 4, m.faces.size(), f) != m.faces.size()) return false;
    for (int i = 0; i < Mv; ++i) {
        int xyz[3];
        if (fread(xyz, 4, 3, f) != 3) return false;
        m.coords[i] = { xyz[0], xyz[1], xyz[2] };
    }
    if (Mv && fread(m.pbr6.data(), 4, m.pbr6.size(), f) != m.pbr6.size()) return false;
    return true;
}

}  // namespace

bool save_master(const std::string& path, const Master& m) {
    const std::string part = part_path(path);
    FILE* f = fopen(part.c_str(), "wb");
    if (!f) { fprintf(stderr, "      [master] cannot write %s\n", part.c_str()); return false; }

    const int V = m.V(), F = m.F(), Mv = m.Mv();
    bool ok = wr(f, MASTER_MAGIC) && wr(f, MASTER_VERSION)
           && wr(f, m.mesh_res) && wr(f, m.pbr_res)
           && wr(f, V) && wr(f, F) && wr(f, Mv);
    ok = ok && fwrite(m.verts.data(), 4, m.verts.size(), f) == m.verts.size();
    ok = ok && fwrite(m.faces.data(), 4, m.faces.size(), f) == m.faces.size();
    for (int i = 0; ok && i < Mv; ++i) {
        const int xyz[3] = { m.coords[i][0], m.coords[i][1], m.coords[i][2] };
        ok = fwrite(xyz, 4, 3, f) == 3;
    }
    if (ok && Mv) ok = fwrite(m.pbr6.data(), 4, m.pbr6.size(), f) == m.pbr6.size();
    fclose(f);

    if (!ok) {
        fprintf(stderr, "      [master] short write to %s\n", part.c_str());
        discard_part(path);
        return false;
    }
    if (!commit_part(path)) return false;

    const double mb = ((double)m.verts.size() * 4 + (double)m.faces.size() * 4
                     + (double)Mv * 3 * 4 + (double)m.pbr6.size() * 4) / (1024.0 * 1024.0);
    printf("      [master] %s (V=%d F=%d, PBR=%d @res%d, mesh @res%d, %.0f MB)\n",
           path.c_str(), V, F, Mv, m.pbr_res, m.mesh_res, mb);
    fflush(stdout);
    return true;
}

bool load_master(const std::string& path, Master& m) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "[master] cannot open %s\n", path.c_str()); return false; }

    uint32_t head = 0;
    if (!rd(f, head)) { fclose(f); fprintf(stderr, "[master] %s is empty\n", path.c_str()); return false; }

    int V = 0, F = 0, Mv = 0;
    bool ok = false;
    if (head == MASTER_MAGIC) {
        uint32_t ver = 0;
        ok = rd(f, ver);
        if (ok && ver != MASTER_VERSION) {
            fclose(f);
            fprintf(stderr, "[master] %s is format v%u, this build reads v%u\n",
                    path.c_str(), ver, MASTER_VERSION);
            return false;
        }
        ok = ok && rd(f, m.mesh_res) && rd(f, m.pbr_res) && rd(f, V) && rd(f, F) && rd(f, Mv);
        m.legacy = false;
    } else {
        // Legacy headerless layout: the u32 just read is V, and one `res` field
        // stands in for both resolutions because the file cannot distinguish them.
        V = (int)head;
        int res = 0;
        ok = rd(f, F) && rd(f, Mv) && rd(f, res);
        m.mesh_res = res;
        m.pbr_res  = res;
        m.legacy   = true;
        if (ok)
            fprintf(stderr, "[master] %s is the legacy single-resolution layout; assuming the mesh\n"
                            "         and the PBR volume are both res %d. If the PBR volume was\n"
                            "         dropped to 512 against a res-1024 mesh, the remesh grid, weld\n"
                            "         epsilon and band width here are wrong -- regenerate it.\n",
                    path.c_str(), res);
    }

    if (!ok || V <= 0 || F <= 0 || Mv < 0 || m.mesh_res <= 0 || m.pbr_res <= 0) {
        fclose(f);
        fprintf(stderr, "[master] %s: implausible header V=%d F=%d voxels=%d mesh_res=%d pbr_res=%d\n",
                path.c_str(), V, F, Mv, m.mesh_res, m.pbr_res);
        return false;
    }
    ok = read_body(f, m, V, F, Mv);
    fclose(f);
    if (!ok) { fprintf(stderr, "[master] %s: truncated\n", path.c_str()); return false; }

    printf("loaded: V=%d F=%d voxels=%d mesh@res%d pbr@res%d%s\n",
           V, F, Mv, m.mesh_res, m.pbr_res, m.legacy ? "  (legacy layout)" : "");
    return true;
}

}  // namespace trellis
