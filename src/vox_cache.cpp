#include "vox_cache.h"
#include "atomic_file.h"
#include <cstdint>
#include <cstdio>

namespace trellis {

namespace {
// "TVX1" — bumped if the layout changes, so a stale cache is refused rather
// than silently read as garbage.
const uint32_t MAGIC   = 0x31585654u;
const uint32_t VERSION = 2;   // v2 adds the optional LR shape slat + upsampled coords

template <typename T>
bool rd(FILE* f, T& v) { return fread(&v, sizeof(T), 1, f) == 1; }
template <typename T>
bool wr(FILE* f, const T& v) { return fwrite(&v, sizeof(T), 1, f) == 1; }
}  // namespace

bool save_vox_cache(const std::string& path, const VoxCache& v) {
    // Written to <path>.part and renamed on success: a run killed mid-write --
    // which is how a caller normally cancels -- must never leave a truncated
    // cache that a later run reads back as a hit.
    const std::string part = part_path(path);
    FILE* f = fopen(part.c_str(), "wb");
    if (!f) { fprintf(stderr, "      [vox] cannot write %s\n", part.c_str()); return false; }
    const uint32_t nc = (uint32_t)v.coords.size();
    const uint32_t n0 = (uint32_t)v.cond.size();
    const uint32_t n1 = (uint32_t)v.cond1024.size();
    const uint32_t casc = v.cascade ? 1u : 0u;
    const uint32_t nlr = (uint32_t)v.lr_slat.size();
    const uint32_t nhr = (uint32_t)v.hr_coords.size();
    bool ok = wr(f, MAGIC) && wr(f, VERSION) && wr(f, casc)
           && wr(f, (uint32_t)v.hr_res) && wr(f, (uint32_t)v.grid) && wr(f, v.seed)
           && wr(f, nc) && wr(f, n0) && wr(f, n1)
           && wr(f, (uint32_t)v.lr_steps) && wr(f, nlr) && wr(f, nhr);
    for (uint32_t i = 0; ok && i < nc; ++i)
        ok = fwrite(v.coords[i].data(), sizeof(int), 3, f) == 3;
    if (ok && n0) ok = fwrite(v.cond.data(),     sizeof(float), n0, f) == n0;
    if (ok && n1) ok = fwrite(v.cond1024.data(), sizeof(float), n1, f) == n1;
    if (ok && nlr) ok = fwrite(v.lr_slat.data(), sizeof(float), nlr, f) == nlr;
    for (uint32_t i = 0; ok && i < nhr; ++i)
        ok = fwrite(v.hr_coords[i].data(), sizeof(int), 3, f) == 3;
    fclose(f);
    if (!ok) {
        fprintf(stderr, "      [vox] short write to %s\n", part.c_str());
        discard_part(path);
        return false;
    }
    if (!commit_part(path)) return false;
    const double mb = ((double)(n0 + n1 + nlr) * sizeof(float)
                     + (double)nhr * 3 * sizeof(int)) / (1024.0*1024.0);
    printf("      [vox] cache -> %s (%u voxels, cond %d+%d tokens", path.c_str(), nc, v.Lc(), v.Lc1024());
    if (nlr) printf(", LR slat @%d steps + %u upsampled coords", v.lr_steps, nhr);
    printf(", %.1f MB)\n", mb);
    fflush(stdout);
    return true;
}

bool load_vox_cache(const std::string& path, VoxCache& v) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open vox cache %s\n", path.c_str()); return false; }
    uint32_t magic = 0, version = 0, casc = 0, hr = 0, grid = 0, nc = 0, n0 = 0, n1 = 0;
    uint32_t lrs = 0, nlr = 0, nhr = 0;
    bool ok = rd(f, magic) && rd(f, version) && rd(f, casc)
           && rd(f, hr) && rd(f, grid) && rd(f, v.seed)
           && rd(f, nc) && rd(f, n0) && rd(f, n1)
           && rd(f, lrs) && rd(f, nlr) && rd(f, nhr);
    if (ok && magic != MAGIC)     { fprintf(stderr, "%s: not a vox cache\n", path.c_str()); ok = false; }
    if (ok && version != VERSION) { fprintf(stderr, "%s: cache version %u, expected %u; regenerate it\n",
                                            path.c_str(), version, VERSION); ok = false; }
    if (ok && (nc == 0 || n0 == 0 || n0 % 1024 || n1 % 1024)) {
        fprintf(stderr, "%s: implausible cache (%u voxels, cond %u/%u)\n", path.c_str(), nc, n0, n1);
        ok = false;
    }
    if (ok) {
        v.cascade = casc != 0;
        v.hr_res  = (int)hr;
        v.grid    = (int)grid;
        v.coords.resize(nc);
        v.cond.resize(n0);
        v.cond1024.resize(n1);
        for (uint32_t i = 0; ok && i < nc; ++i)
            ok = fread(v.coords[i].data(), sizeof(int), 3, f) == 3;
        if (ok && n0) ok = fread(v.cond.data(),     sizeof(float), n0, f) == n0;
        if (ok && n1) ok = fread(v.cond1024.data(), sizeof(float), n1, f) == n1;
        v.lr_steps = (int)lrs;
        v.lr_slat.resize(nlr);
        v.hr_coords.resize(nhr);
        if (ok && nlr) ok = fread(v.lr_slat.data(), sizeof(float), nlr, f) == nlr;
        for (uint32_t i = 0; ok && i < nhr; ++i)
            ok = fread(v.hr_coords[i].data(), sizeof(int), 3, f) == 3;
        if (!ok) fprintf(stderr, "%s: truncated cache\n", path.c_str());
    }
    fclose(f);
    if (!ok) return false;
    printf("      [vox] resumed from %s: %u voxels @grid %d, cond %d+%d tokens, seed %u%s",
           path.c_str(), nc, v.grid, v.Lc(), v.Lc1024(), v.seed,
           v.cascade ? ", cascade" : "");
    if (v.lr_steps) printf(", LR slat @%d steps + %u upsampled coords", v.lr_steps, nhr);
    printf("\n");
    fflush(stdout);
    return true;
}

}  // namespace trellis
