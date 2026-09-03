#include "remesh_stage.h"
#include "mesh_audit.h"
#include <algorithm>
#include <chrono>
#include <cstdio>

namespace trellis {

namespace {
double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace

// Sort the packed edge keys rather than hashing them, matching what the mesh
// passes were changed to for the same reason: with millions of distinct keys a
// hash map is all cache misses. Same definition as before -- an undirected edge
// used exactly once is a boundary -- so the number is unchanged.
double open_per_1k(const std::vector<int32_t>& faces) {
    const size_t F = faces.size() / 3;
    if (F == 0) return 1e9;
    std::vector<uint64_t> e;
    e.reserve(F * 3);
    for (size_t f = 0; f < F; ++f)
        for (int j = 0; j < 3; ++j) {
            int a = faces[3*f+j], b = faces[3*f+(j+1)%3];
            if (a > b) { int t = a; a = b; b = t; }
            e.push_back(((uint64_t)(uint32_t)a << 32) | (uint32_t)b);
        }
    std::sort(e.begin(), e.end());
    size_t nb = 0;
    for (size_t i = 0; i < e.size();) {
        size_t j = i + 1;
        while (j < e.size() && e[j] == e[i]) ++j;
        if (j - i == 1) ++nb;
        i = j;
    }
    return 1000.0 * (double)nb / (double)F;
}

RemeshStageResult remesh_stage(const std::vector<float>& verts,
                               const std::vector<int32_t>& faces,
                               const TriBvh& bvh,
                               const RemeshStageOpts& o,
                               bool want_strip_in) {
    RemeshStageResult r;
    r.final_mode = o.mode;
    r.want_strip = want_strip_in;

    const int band = o.band > 0 ? o.band : std::max(1, o.res / 512);
    double t = now_s();

    // Steps 3-6. Returns open edges per 1000 faces on the finished high-poly,
    // which is what the `auto` gate reads.
    auto build = [&](RemeshMode m) -> double {
        r.mesh = remesh_narrow_band_dc(verts.data(), (int64_t)verts.size()/3,
                                       faces.data(), (int64_t)faces.size()/3,
                                       bvh, o.res, band, o.project, m,
                                       o.sign_rays, o.sign_dump, o.parity_coarse);
        if (o.verbose) { printf("  [remesh %.1fs]\n", now_s()-t); t = now_s(); }
        if (o.on_stage) o.on_stage("remesh", r.mesh.faces);
        if (r.mesh.F() == 0) return 1e9;
        clean_mesh(r.mesh.V(), r.mesh.faces);
        if (o.on_stage) o.on_stage("clean_mesh", r.mesh.faces);
        const int ndrop = drop_small_components(r.mesh.verts, r.mesh.faces, 0.02f);
        printf("  remesh postproc: dropped %d floater comps -> V=%d F=%d\n",
               ndrop, r.mesh.V(), r.mesh.F());
        fflush(stdout);
        if (o.verbose) { printf("  [clean+drop %.1fs] dropped=%d\n", now_s()-t, ndrop); t = now_s(); }
        if (o.on_stage) o.on_stage("drop_components", r.mesh.faces);
        if (m == RemeshMode::Unsigned) return open_per_1k(r.mesh.faces);
        // Round holes wider than the band's own lip survive contouring as clean
        // rims; fan them now so QEM is never forced to preserve rim vertices and
        // the normal bake never sees a hole. (Slits narrower than 2*eps sealed
        // themselves under the lip, which is why unsigned never needed this.)
        if (o.fill_hipoly > 0.0f) {
            const int nfill = fill_holes(r.mesh.verts, r.mesh.faces, o.fill_hipoly);
            if (nfill) {
                printf("  remesh postproc: fan-filled %d high-poly holes\n", nfill);
                fflush(stdout);
            }
            if (o.verbose) { printf("  [fill_holes %.1fs] filled=%d (max perimeter %.3f)\n",
                                    now_s()-t, nfill, o.fill_hipoly); t = now_s(); }
            if (o.on_stage) o.on_stage("fill_holes", r.mesh.faces);
        }
        // Meaningful only on a closed output: its cavities are genuinely
        // enclosed, rather than being the exterior seen from between two sheets
        // of a double cover.
        if (o.cull) {
            cull_enclosed_components(r.mesh.verts, r.mesh.faces, 256);
            if (o.verbose) { printf("  [cull %.1fs]\n", now_s()-t); t = now_s(); }
            if (o.on_stage) o.on_stage("cull", r.mesh.faces);
        }
        return open_per_1k(r.mesh.faces);
    };

    r.open1k = build(o.mode);

    // The gate the design puts on the OUTPUT rather than the input: Interior's
    // only failure modes are closed artefacts, so the output states directly
    // what an input-side router would have to guess at.
    if (o.mode_auto && o.mode != RemeshMode::Unsigned) {
        const size_t nf = r.mesh.faces.size() / 3;
        if (nf < 10000 || r.open1k > 5.0) {
            printf("  remesh: interior mode failed (faces=%zu, open/1k=%.2f); "
                   "falling back to unsigned + strip\n", nf, r.open1k);
            fflush(stdout);
            r.mesh = Mesh();
            r.open1k     = build(RemeshMode::Unsigned);
            r.final_mode = RemeshMode::Unsigned;
            r.fell_back  = true;
            r.want_strip = true;   // the unsigned shell is double-covered
        }
    }
    return r;
}

}  // namespace trellis
