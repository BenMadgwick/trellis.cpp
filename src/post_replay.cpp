// post-replay — re-run the post-neural stages (weld, hole fill, decimation, UV
// bake, GLB write) from a TRELLIS_DUMP_POST dump, skipping the ~10-minute
// neural pipeline. Development harness for iterating on mesh/texture
// post-processing.
//
//   post-replay <dump.bin> <out.glb> [--box-uv] [--faces N] [--atlas T]
//               [--decim GRID] [--no-weld] [--no-fill]
//               [--save-mesh F] [--load-mesh F]         cache after the remesh
//               [--save-stripped F] [--load-stripped F] cache after the strip
//               [--faces N[,N,...]]                     one GLB per target
//               [--normal-map] [--normal-space tangent|object] [--normal-search K]
//               [--remesh-mode unsigned|signed5|interior|auto]  which field to contour
//               [--sign-rays N] [--sign-dump F] [--no-cull] [--fill-hipoly P]
//               [--remesh-project X] [--audit-visible]
#include "uv_bake.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "remesh_stage.h"
#include "mesh_audit.h"
#include "strip_interior.h"
#include "shading.h"
#include "mesh_glb.h"
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using trellis::VoxelPbr;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// boundary/non-manifold audit over the welded index space (positions assumed welded)
#include <unordered_map>
static size_t audit(const char* tag, const std::vector<int32_t>& faces) {
    // count, plus a direction balance: each use of the edge contributes +1 when
    // traversed low->high and -1 for high->low. Two faces sharing an edge are
    // consistently wound exactly when they traverse it in OPPOSITE directions,
    // i.e. balance 0. This is the A6 quantity, measured directly rather than
    // inferred from how many faces clean_mesh's BFS decided to flip (that count
    // is relative to an arbitrary per-patch seed, so on a mesh cut into many
    // patches by boundary edges it lands near 50% whatever the winding is).
    struct E { int count, bal; };
    std::unordered_map<uint64_t,E> e;
    e.reserve(faces.size() * 2);
    const size_t F = faces.size() / 3;
    auto k = [](int a, int b){ if (a>b){int t=a;a=b;b=t;} return ((uint64_t)(uint32_t)a<<32)|(uint32_t)b; };
    for (size_t f = 0; f < F; ++f)
        for (int j = 0; j < 3; ++j) {
            const int a = faces[3*f+j], b = faces[3*f+(j+1)%3];
            E& v = e[k(a, b)];
            ++v.count;
            v.bal += a < b ? 1 : -1;
        }
    size_t nb = 0, nm = 0, n2 = 0, cons = 0;
    for (auto& kv : e) {
        if (kv.second.count == 1) ++nb;
        else if (kv.second.count > 2) ++nm;
        else { ++n2; if (kv.second.bal == 0) ++cons; }
    }
    printf("  [audit] %-22s F=%-9zu boundary_edges=%-7zu nonmanifold=%zu",
           tag, F, nb, nm);
    if (F) printf("  open/1k=%.2f", 1000.0 * (double)nb / (double)F);
    if (n2) printf("  wound=%.2f%%", 100.0 * (double)cons / (double)n2);
    printf("\n");
    fflush(stdout);
    return nb;
}

static void audit_visible(const char* tag, const std::vector<float>& verts,
                          const std::vector<int32_t>& faces, int ndirs, int grid) {
    const trellis::VisibleAudit a = trellis::visible_fraction(verts, faces, ndirs, grid);
    printf("  [visible] %-20s faces %lld/%lld = %.4f   area %.4f", tag,
           (long long)a.faces_hit, (long long)a.faces, a.face_frac(), a.area_frac());
    // A ray marks at most one face, so on a mesh with millions of faces this
    // saturates on the ray budget and reports rays/faces, not visibility. Say
    // so rather than letting a low number read as "most of it is buried".
    if (a.undersampled())
        printf("   [LOWER BOUND: %lld rays landed on %lld faces, %.1f per face; raise --audit-grid]",
               (long long)a.rays_hit, (long long)a.faces_hit,
               a.faces_hit ? (double)a.rays_hit / (double)a.faces_hit : 0.0);
    printf("\n");
    fflush(stdout);
}

// Mesh cache blobs: int V, int F, float verts[V*3], int32 faces[F*3]. Same
// layout TRELLIS_DUMP_DECMESH writes, so the two are interchangeable. Used for
// the post-remesh and post-strip caches, which between them skip ~4.5 min of
// the pipeline when iterating on face targets.
static bool load_mesh_bin(const char* path, std::vector<float>& verts,
                          std::vector<int32_t>& faces, const char* tag) {
    FILE* mf = fopen(path, "rb");
    if (!mf) return false;
    bool ok = false;
    int mV = 0, mF = 0;
    if (fread(&mV,4,1,mf) == 1 && fread(&mF,4,1,mf) == 1 && mV > 0 && mF > 0) {
        verts.resize((size_t)mV*3);
        faces.resize((size_t)mF*3);
        ok = fread(verts.data(),4,verts.size(),mf) == verts.size()
          && fread(faces.data(),4,faces.size(),mf) == faces.size();
        if (ok) printf("  [cache] loaded %s V=%d F=%d from %s\n", tag, mV, mF, path);
    }
    fclose(mf);
    if (!ok) { verts.clear(); faces.clear(); fprintf(stderr, "  [cache] %s unusable, recomputing\n", path); }
    return ok;
}

static void save_mesh_bin(const char* path, const std::vector<float>& verts,
                          const std::vector<int32_t>& faces, const char* tag) {
    FILE* mf = fopen(path, "wb");
    if (!mf) { fprintf(stderr, "  [cache] cannot write %s\n", path); return; }
    const int mV = (int)(verts.size()/3), mF = (int)(faces.size()/3);
    fwrite(&mV,4,1,mf); fwrite(&mF,4,1,mf);
    fwrite(verts.data(),4,verts.size(),mf);
    fwrite(faces.data(),4,faces.size(),mf);
    fclose(mf);
    printf("  [cache] saved %s V=%d F=%d -> %s\n", tag, mV, mF, path);
}

// Output path for one tier of a sweep: "bear.glb" + 2500 -> "bear_2500.glb".
// A single-target run keeps the exact path it was given, so existing callers and
// scripts see no change.
static std::string tier_path(const char* base, int faces, bool suffix) {
    std::string b(base);
    if (!suffix) return b;
    const size_t dot = b.find_last_of('.');
    const size_t sep = b.find_last_of("/\\");
    const bool has_ext = dot != std::string::npos && (sep == std::string::npos || dot > sep);
    char n[32];
    std::snprintf(n, sizeof(n), "_%d", faces);
    return has_ext ? b.substr(0, dot) + n + b.substr(dot) : b + n;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: post-replay <dump.bin> <out.glb> [opts]\n"); return 1; }
    const char* dump = argv[1];
    const char* out = argv[2];
    bool boxuv = false, do_weld = true, do_fill = true, do_bake = true, do_remesh = true, do_snap = true;
    int band = 1;
    bool do_strip = false;
    trellis::StripOpts strip;
    int atlas = 2048, decim = -1;
    // One or more QEM face targets. A sweep shares every stage before
    // decimation -- the strip, both BVHs, the high-poly normals -- so the
    // marginal cost of an extra tier is just decimate + bake.
    std::vector<int> targets;
    const char* save_mesh = nullptr; const char* load_mesh = nullptr;
    const char* save_stripped = nullptr; const char* load_stripped = nullptr;
    // Which field remesh_dc contours (see RemeshMode in remesh_dc.h). `auto` is
    // Interior plus the output-audit fallback below; `interior` is Interior with
    // no fallback, for experiments that need the failure to be visible.
    trellis::RemeshMode rmode = trellis::RemeshMode::Interior;
    bool mode_auto = true;
    int sign_rays = 64;
    int parity_coarse = 2;
    const char* sign_dump = nullptr;
    bool do_cull = true;
    // Perimeter ceiling for the fan-fill on the HIGH-POLY. Holes wider than the
    // band's own lip (2*eps ~ 3.9 mm) survive contouring as clean rims -- the
    // dog's three paw holes are ~150 mm of perimeter each. Filling them here
    // rather than after decimation means QEM is never forced to preserve rim
    // vertices and the normal bake never sees a hole.
    float fill_hipoly = 0.25f;
    float remesh_project = 0.0f;
    bool do_visible = false;
    int vis_dirs = 512, vis_grid = 128;
    // Contouring grid resolution, independent of the dump's own res. The UDF is
    // evaluated from the mesh through a BVH, so a finer grid is simply more
    // samples of the same field -- and thin features are lost when the region
    // between two surfaces is thinner than a cell, which is the suspected cause
    // of the tearing on hollow reconstructions.
    int remesh_res = 0;
    // Largest boundary loop fill_small_holes will close. The default of 64 is
    // fine for the unsigned path, which never asks what is inside; ray parity
    // does, and a ray that escapes through a surviving hole miscounts. Input
    // boundary edges track the tearing almost exactly: dog 3,756 -> 0.2 open
    // edges per 1000 faces, table 121,631 -> 9.9, jerry can 128,213 -> 17.1.
    int fill_loop = 64;
    bool do_normal = false, nrm_tangent = true;
    float nrm_search = 2.0f;
    float nrm_min_dot = 0.85f;
    // Search ceiling as a multiple of the offset shell's own thickness (2*eps).
    // Expressed in shell thicknesses because that is the thing being tunnelled
    // through, and it is known exactly rather than guessed.
    float nrm_cap_shells = 4.0f;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--box-uv") boxuv = true;
        else if (a == "--faces" && i+1 < argc) {
            const std::string list = argv[++i];
            size_t b = 0;
            while (b <= list.size()) {
                const size_t e = list.find(',', b);
                const std::string tok = list.substr(b, e == std::string::npos ? std::string::npos : e - b);
                if (!tok.empty()) {
                    const int v = atoi(tok.c_str());
                    if (v > 0) targets.push_back(v);
                    else fprintf(stderr, "--faces: ignoring non-positive target '%s'\n", tok.c_str());
                }
                if (e == std::string::npos) break;
                b = e + 1;
            }
        }
        else if (a == "--atlas" && i+1 < argc) atlas = atoi(argv[++i]);
        else if (a == "--decim" && i+1 < argc) decim = atoi(argv[++i]);
        else if (a == "--no-weld") do_weld = false;
        else if (a == "--no-fill") do_fill = false;
        else if (a == "--no-bake") do_bake = false;
        else if (a == "--no-remesh") do_remesh = false;
        else if (a == "--band" && i+1 < argc) band = atoi(argv[++i]);
        else if (a == "--no-snap") do_snap = false;
        else if (a == "--signed-remesh") { rmode = trellis::RemeshMode::Signed5; mode_auto = false; }
        else if (a == "--remesh-mode" && i+1 < argc) {
            const std::string m = argv[++i];
            mode_auto = m == "auto";
            if (m == "unsigned")      rmode = trellis::RemeshMode::Unsigned;
            else if (m == "signed5")  rmode = trellis::RemeshMode::Signed5;
            else if (m == "interior" || m == "auto") rmode = trellis::RemeshMode::Interior;
            else { fprintf(stderr, "--remesh-mode: expected unsigned|signed5|interior|auto, got '%s'\n", m.c_str()); return 1; }
        }
        else if (a == "--sign-rays" && i+1 < argc) sign_rays = atoi(argv[++i]);
        else if (a == "--parity-coarse" && i+1 < argc) parity_coarse = atoi(argv[++i]);
        else if (a == "--sign-dump" && i+1 < argc) sign_dump = argv[++i];
        else if (a == "--no-cull") do_cull = false;
        else if (a == "--fill-hipoly" && i+1 < argc) fill_hipoly = (float)atof(argv[++i]);
        else if (a == "--remesh-project" && i+1 < argc) remesh_project = (float)atof(argv[++i]);
        else if (a == "--audit-visible") do_visible = true;
        else if (a == "--audit-dirs" && i+1 < argc) vis_dirs = atoi(argv[++i]);
        else if (a == "--audit-grid" && i+1 < argc) vis_grid = atoi(argv[++i]);
        else if (a == "--remesh-res" && i+1 < argc) remesh_res = atoi(argv[++i]);
        else if (a == "--fill-loop" && i+1 < argc) fill_loop = atoi(argv[++i]);
        else if (a == "--strip-interior") do_strip = true;
        else if (a == "--strip-grid" && i+1 < argc) strip.grid = atoi(argv[++i]);
        else if (a == "--strip-depth" && i+1 < argc) strip.depth = atoi(argv[++i]);
        else if (a == "--strip-seal" && i+1 < argc) strip.seal = atoi(argv[++i]);
        else if (a == "--strip-outward") strip.outward = true;
        else if (a == "--strip-smooth" && i+1 < argc) strip.smooth = atoi(argv[++i]);
        else if (a == "--strip-no-shrink") strip.shrink = false;
        else if (a == "--strip-fold" && i+1 < argc) strip.fold_deg = (float)atof(argv[++i]);
        else if (a == "--strip-shrink-iters" && i+1 < argc) strip.shrink_iters = atoi(argv[++i]);
        else if (a == "--strip-shrink-beta" && i+1 < argc) strip.shrink_beta = (float)atof(argv[++i]);
        else if (a == "--save-mesh" && i+1 < argc) save_mesh = argv[++i];
        else if (a == "--load-mesh" && i+1 < argc) load_mesh = argv[++i];
        else if (a == "--save-stripped" && i+1 < argc) save_stripped = argv[++i];
        else if (a == "--load-stripped" && i+1 < argc) load_stripped = argv[++i];
        else if (a == "--normal-map") do_normal = true;
        else if (a == "--normal-space" && i+1 < argc) nrm_tangent = std::string(argv[++i]) != "object";
        else if (a == "--normal-search" && i+1 < argc) nrm_search = (float)atof(argv[++i]);
        else if (a == "--normal-min-dot" && i+1 < argc) nrm_min_dot = (float)atof(argv[++i]);
        else if (a == "--normal-cap-shells" && i+1 < argc) nrm_cap_shells = (float)atof(argv[++i]);
    }

    if (targets.empty()) targets.push_back(300000);

    // A normal bake against the unstripped shell samples buried, inward-facing
    // sheets over most of the surface; the result is noise that looks plausible
    // in the texture viewer. Refuse rather than produce it.
    if (do_normal && !do_strip && !load_stripped && rmode == trellis::RemeshMode::Unsigned) {
        fprintf(stderr, "--normal-map requires --strip-interior (or --load-stripped): baking against\n"
                        "the four-sheet shell samples buried geometry and yields noise.\n"
                        "--remesh-mode interior|auto|signed5 also satisfies this: none of them\n"
                        "builds the cover in the first place.\n");
        return 1;
    }

    FILE* f = fopen(dump, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", dump); return 1; }
    int V = 0, F = 0, Mv = 0, res = 0;
    // Read the four header fields as separate statements. Summing four fread
    // calls in one expression leaves them unsequenced, so a compiler is free to
    // run them right-to-left (MSVC does) and the fields land in the wrong
    // variables - the sum is still 4, so the check passes and the next read
    // asks for the wrong byte count.
    const bool hdr = fread(&V,4,1,f) == 1 && fread(&F,4,1,f) == 1
                  && fread(&Mv,4,1,f) == 1 && fread(&res,4,1,f) == 1;
    if (!hdr) { fprintf(stderr, "%s: truncated header\n", dump); fclose(f); return 1; }
    if (V <= 0 || F <= 0 || Mv < 0 || res <= 0) {
        fprintf(stderr, "%s: implausible header V=%d F=%d voxels=%d res=%d\n", dump, V, F, Mv, res);
        fclose(f); return 1;
    }
    std::vector<float> verts((size_t)V*3);
    std::vector<int32_t> faces((size_t)F*3);
    std::vector<std::array<int,3>> coords((size_t)Mv);
    std::vector<float> pbr6((size_t)Mv*6);
    if (fread(verts.data(),4,verts.size(),f) != verts.size()) { fprintf(stderr, "%s: short read (verts)\n", dump); fclose(f); return 1; }
    if (fread(faces.data(),4,faces.size(),f) != faces.size()) { fprintf(stderr, "%s: short read (faces)\n", dump); fclose(f); return 1; }
    for (auto& c : coords) if (fread(c.data(),4,3,f) != 3) { fprintf(stderr, "%s: short read (voxel coords)\n", dump); fclose(f); return 1; }
    if (fread(pbr6.data(),4,pbr6.size(),f) != pbr6.size()) { fprintf(stderr, "%s: short read (pbr)\n", dump); fclose(f); return 1; }
    fclose(f);
    printf("loaded: V=%d F=%d voxels=%d res=%d\n", V, F, Mv, res);
    {
        // Volume/area of the INPUT, before any remeshing. Says whether the
        // decoded mesh is already two-walled (ratio ~ half the wall separation)
        // or a genuine single sheet (ratio ~ 0). That decides whether the
        // double cover is created by the decode or entirely by remesh_dc.
        double vol = 0.0, area = 0.0;
        for (int64_t f = 0; f < (int64_t)F; ++f) {
            const float* A = &verts[3*(size_t)faces[3*f+0]];
            const float* B = &verts[3*(size_t)faces[3*f+1]];
            const float* C = &verts[3*(size_t)faces[3*f+2]];
            vol += ((double)A[0]*((double)B[1]*C[2] - (double)B[2]*C[1])
                  - (double)A[1]*((double)B[0]*C[2] - (double)B[2]*C[0])
                  + (double)A[2]*((double)B[0]*C[1] - (double)B[1]*C[0])) / 6.0;
            const double u[3] = {B[0]-A[0], B[1]-A[1], B[2]-A[2]};
            const double v2[3] = {C[0]-A[0], C[1]-A[1], C[2]-A[2]};
            const double cx = u[1]*v2[2]-u[2]*v2[1], cy = u[2]*v2[0]-u[0]*v2[2], cz = u[0]*v2[1]-u[1]*v2[0];
            area += 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
        }
        printf("  [input] signed volume %.2f cm3, area %.1f cm2, vol/area %.4f mm\n",
               vol * 1e6, area * 1e4, (area > 0 ? std::fabs(vol)/area : 0.0) * 1000.0);
        fflush(stdout);
    }

    double t = now();
    if (do_weld) trellis::weld_vertices(verts, faces, nullptr, 1.0f / ((float)res * 8.0f));
    printf("  [weld %.1fs]\n", now()-t); t = now();
    audit("weld", faces);
    if (do_fill) trellis::fill_small_holes(faces, fill_loop);
    printf("  [fill %.1fs]\n", now()-t); t = now();
    audit("fill_small_holes", faces);

    // Step 0. A coincident face pair is two crossings on every ray that meets
    // it, so parity reads solid geometry as a hole. Must precede the BVH build.
    if (rmode == trellis::RemeshMode::Interior || rmode == trellis::RemeshMode::Signed5) {
        const int ndup = trellis::drop_duplicate_faces(faces);
        printf("  [dedupe %.1fs] removed=%d\n", now()-t, ndup); t = now();
        if (ndup) audit("drop_duplicate_faces", faces);
    }

    trellis::TriBvh bvh = trellis::TriBvh::build(verts.data(), (int64_t)verts.size()/3,
                                                 faces.data(), (int64_t)faces.size()/3);
    printf("  [bvh %.1fs]\n", now()-t); t = now();
    trellis::Mesh rm;
    // Dev aid: two resume points. --load-stripped picks up AFTER the strip,
    // skipping remesh+clean+strip (~4.5 min) - the one to use when sweeping
    // face targets, since everything it skips is independent of the target.
    // --load-mesh picks up after the remesh only, which is what strip-parameter
    // experiments need. Neither can skip weld+fill+bvh (~72 s): the bake's
    // albedo snap samples a BVH over the pre-remesh mesh.
    bool cached = false, pre_stripped = false;
    if (load_stripped) {
        pre_stripped = load_mesh_bin(load_stripped, rm.verts, rm.faces, "post-strip");
        cached = pre_stripped;
    }
    if (!cached && load_mesh) cached = load_mesh_bin(load_mesh, rm.verts, rm.faces, "post-remesh");
    if (do_remesh && !cached) {
        const int rres = remesh_res > 0 ? remesh_res : res;
        if (rres != res) printf("  remesh at grid %d (dump res %d)\n", rres, res);
        // Steps 3-6 and the `auto` gate now live in remesh_stage(), shared with
        // trellis-cli. This file had its own copy of both, reaching the same
        // thresholds through a different open-edge count -- and its fallback
        // left `rmode` stale, so the single-cover checks below could still read
        // Interior after having built Unsigned.
        trellis::RemeshStageOpts ro;
        ro.res           = rres;
        ro.band          = band;   // NOTE: defaults to 1 here and to 0 (auto ->
                                   // res/512, i.e. 2 at res 1024) in trellis-cli,
                                   // so a default replay contours a thinner shell
                                   // than the run it replays. Preserved, not fixed.
        ro.project       = remesh_project;
        ro.mode          = rmode;
        ro.mode_auto     = mode_auto;
        ro.sign_rays     = sign_rays;
        ro.parity_coarse = parity_coarse;
        ro.cull          = do_cull;
        ro.fill_hipoly   = fill_hipoly;
        ro.sign_dump     = sign_dump;
        ro.verbose       = true;   // this is the harness: keep the per-stage timings
        ro.on_stage      = [](const char* tag, const std::vector<int32_t>& f) { audit(tag, f); };
        trellis::RemeshStageResult rs = trellis::remesh_stage(verts, faces, bvh, ro, do_strip);
        rm       = std::move(rs.mesh);
        rmode    = rs.final_mode;
        do_strip = rs.want_strip;
        t = now();
    }
    std::vector<float>& sverts = rm.F() > 0 ? rm.verts : verts;
    std::vector<int32_t>& sfaces = rm.F() > 0 ? rm.faces : faces;

    if (save_mesh && !cached) save_mesh_bin(save_mesh, sverts, sfaces, "post-remesh mesh");

    // Opt-in divergence from the reference: drop the buried sheets of the
    // narrow-band shell before simplifying. The offset surface is two-sided by
    // construction and the input was already two-walled, so most of what QEM is
    // about to spend its budget on can never be seen - and being crumpled, it
    // scores as high-curvature detail and outbids the smooth visible skin.
    if (do_strip && !pre_stripped) {
        // The pre-remesh mesh and its BVH, which remesh_dc already built.
        strip.ref_verts = verts.data(); strip.ref_faces = faces.data();
        strip.ref_V = (int64_t)verts.size()/3; strip.ref_F = (int64_t)faces.size()/3;
        strip.ref_bvh = &bvh;
        trellis::strip_interior(sverts, sfaces, strip);
        audit("strip_interior", sfaces);
        printf("  [strip %.1fs]\n", now()-t); t = now();
    } else if (pre_stripped) {
        audit("loaded post-strip", sfaces);
    }
    if (save_stripped && !pre_stripped) {
        if (!do_strip) fprintf(stderr, "  [cache] --save-stripped without --strip-interior: saving an UNSTRIPPED mesh\n");
        save_mesh_bin(save_stripped, sverts, sfaces, "post-strip mesh");
    }

    // Triangle-budget efficiency of the thing we are about to spend the budget
    // on. Run before the --no-bake early return, so a measurement-only sweep
    // (which is exactly how this number gets collected) still reports it.
    if (do_visible) { audit_visible("high-poly", sverts, sfaces, vis_dirs, vis_grid); t = now(); }

    if (!do_bake && targets.size() <= 1) {
        // nothing past this point is needed when the run exists only to populate caches
        std::vector<float> dv, dp; std::vector<int32_t> df;
        if (decim > 0) trellis::decimate_cluster(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, {}, decim, dv, df, dp);
        else if (decim == 0) { dv = sverts; df = sfaces; }
        else trellis::decimate_qem(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, targets[0], dv, df);
        printf("  [decimate %.1fs]\n", now()-t); t = now();
        if (do_visible) {
            char tag[32];
            std::snprintf(tag, sizeof(tag), "tier %d", targets[0]);
            audit_visible(tag, dv, df, vis_dirs, vis_grid);
        }
        printf("(--no-bake) done\n");
        return 0;
    }

    VoxelPbr vox{&coords, &pbr6, res, do_snap ? &bvh : nullptr};
    const std::vector<float> no_vp;

    // High-poly source for the normal bake: the stripped shell, plus its own
    // BVH and vertex normals. Distinct from `bvh` above, which is over the
    // PRE-remesh two-walled mesh -- fine for the albedo snap (position only)
    // and useless here, where the sheet you land on decides the answer.
    //
    // Built ONCE, outside the tier loop. Nothing up to this point depends on the
    // face target, so a sweep pays for it all exactly once: ~72 s of
    // weld/fill/bvh plus ~3 s here, against ~5 s of decimation and ~15 s of bake
    // per tier. That ratio is the whole reason sweeping tiers is cheap.
    trellis::NormalSrc nsrc;
    trellis::TriBvh hi_bvh;
    std::vector<float> hi_nrm;
    if (do_normal) {
        hi_bvh = trellis::TriBvh::build(sverts.data(), (int64_t)sverts.size()/3,
                                        sfaces.data(), (int64_t)sfaces.size()/3);
        printf("  [hi-bvh %.1fs] over %zu faces\n", now()-t, sfaces.size()/3); t = now();
        trellis::vertex_normals(sverts.data(), (int64_t)sverts.size()/3,
                                sfaces.data(), (int64_t)sfaces.size()/3, hi_nrm);
        printf("  [hi-normals %.1fs]\n", now()-t); t = now();
        nsrc.verts = sverts.data(); nsrc.faces = sfaces.data(); nsrc.vnrm = hi_nrm.data();
        nsrc.V = (int64_t)sverts.size()/3; nsrc.F = (int64_t)sfaces.size()/3;
        nsrc.bvh = &hi_bvh;
        nsrc.tangent_space = nrm_tangent;
        nsrc.search_scale = nrm_search;
        nsrc.min_consensus = nrm_min_dot;
        // eps = band*scale/res with scale = (res + 3*band)/res -- the same
        // expression remesh_dc uses, so the ceiling tracks whatever band the
        // run actually used rather than assuming the default.
        {
            const float scale = (res + 3.f * band) / (float)res;
            const float eps = band * scale / (float)res;
            if (nrm_cap_shells > 0.f) nsrc.search_cap = nrm_cap_shells * 2.f * eps;
        }
    }

    int failures = 0;
    for (size_t ti = 0; ti < targets.size(); ++ti) {
        const int target = targets[ti];
        const std::string outpath = tier_path(out, target, targets.size() > 1);
        if (targets.size() > 1) printf("\n=== tier: %d faces -> %s ===\n", target, outpath.c_str());
        t = now();

        std::vector<float> dv, dp; std::vector<int32_t> df;
        if (decim > 0) trellis::decimate_cluster(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, {}, decim, dv, df, dp);
        else if (decim == 0) { dv = sverts; df = sfaces; }
        else {
            // match the CLI: faithful QEM port (not the old meshopt/FQMS decimate_simplify)
            trellis::decimate_qem(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, target, dv, df);
            audit("decimate_qem", df);
            trellis::weld_vertices(dv, df, nullptr, 1.0f / ((float)res * 8.0f));
            trellis::fill_small_holes(df);
            int ndrop2 = trellis::drop_small_components(dv, df, 0.03f);
            if (ndrop2) printf("  dropped %d more comps\n", ndrop2);
            audit("post-decimate", df);
        }
        printf("  [decimate %.1fs]\n", now()-t); t = now();
        if (do_visible) {
            char tag[32];
            std::snprintf(tag, sizeof(tag), "tier %d", target);
            audit_visible(tag, dv, df, vis_dirs, vis_grid);
            t = now();
        }
        // Geometry-only sweep: the per-tier audits above are the A5/A6 evidence
        // and cost seconds, while a bake costs minutes. Without this, --no-bake
        // only did anything for a single target.
        if (!do_bake) continue;

        trellis::BakedMesh bm = boxuv
            ? trellis::uv_box_project(dv, (int)dv.size()/3, df, (int)df.size()/3, no_vp, atlas, &vox)
            : trellis::uv_bake(dv, (int)dv.size()/3, df, (int)df.size()/3, no_vp, atlas, &vox,
                               do_normal ? &nsrc : nullptr);
        if (!boxuv && !bm.ok())
            bm = trellis::uv_chart_project(dv, (int)dv.size()/3, df, (int)df.size()/3, no_vp, atlas, &vox);
        printf("  [bake %.1fs]\n", now()-t);
        if (!bm.ok()) {
            // One bad tier must not throw away the tiers that did work.
            fprintf(stderr, "bake failed for target %d\n", target);
            ++failures;
            continue;
        }
        printf("  [audit] bake: faces in=%zu out=%zu (dropped %lld)\n",
               df.size()/3, bm.faces.size()/3, (long long)(df.size()/3) - (long long)(bm.faces.size()/3));

        // Only a tangent-space map goes into the GLB: glTF's normalTexture is
        // defined as tangent space, so an object-space bake stays a debug
        // artifact (written, but as the plain textured GLB it would otherwise be).
        const bool ship_nmap = bm.has_normal_map() && bm.nrm_tangent_space;
        const bool wrote = trellis::write_glb_textured(
                                    outpath.c_str(), bm.verts.data(), (int64_t)bm.verts.size()/3, bm.uv.data(),
                                    bm.faces.data(), (int64_t)bm.faces.size()/3, bm.base.data(), bm.mr.data(), bm.T,
                                    /*double_sided=*/rm.F() == 0, -1, nullptr, true,
                                    ship_nmap ? bm.nrm.data() : nullptr,
                                    ship_nmap ? bm.vnrm.data() : nullptr,
                                    ship_nmap ? bm.vtan.data() : nullptr);
        if (bm.has_normal_map() && !bm.nrm_tangent_space)
            printf("  note: object-space map baked for inspection but NOT embedded (glTF normalTexture is tangent space)\n");
        // The writer's return value was previously discarded, so a run whose
        // output directory did not exist reported success and produced nothing.
        // Harmless for one file; in a sweep it loses the whole batch silently.
        if (!wrote) {
            fprintf(stderr, "FAILED to write %s (does its directory exist?)\n", outpath.c_str());
            ++failures;
            continue;
        }
        printf("wrote %s (atlas %d)\n", outpath.c_str(), bm.T);
    }
    return failures ? 1 : 0;
}
