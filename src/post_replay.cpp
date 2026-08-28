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
#include "uv_bake.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
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
static void audit(const char* tag, const std::vector<int32_t>& faces) {
    std::unordered_map<uint64_t,int> e;
    e.reserve(faces.size() * 2);
    const size_t F = faces.size() / 3;
    auto k = [](int a, int b){ if (a>b){int t=a;a=b;b=t;} return ((uint64_t)(uint32_t)a<<32)|(uint32_t)b; };
    for (size_t f = 0; f < F; ++f)
        for (int j = 0; j < 3; ++j) e[k(faces[3*f+j], faces[3*f+(j+1)%3])]++;
    size_t nb = 0, nm = 0;
    for (auto& kv : e) { if (kv.second == 1) ++nb; else if (kv.second > 2) ++nm; }
    printf("  [audit] %-22s F=%-9zu boundary_edges=%-7zu nonmanifold=%zu\n", tag, F, nb, nm);
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
    bool signed_remesh = false;
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
        else if (a == "--signed-remesh") signed_remesh = true;
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
    if (do_normal && !do_strip && !load_stripped && !signed_remesh) {
        fprintf(stderr, "--normal-map requires --strip-interior (or --load-stripped): baking against\n"
                        "the four-sheet shell samples buried geometry and yields noise.\n"
                        "--signed-remesh also satisfies this: it never builds the cover.\n");
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
    if (do_fill) trellis::fill_small_holes(faces);
    printf("  [fill %.1fs]\n", now()-t); t = now();
    audit("fill_small_holes", faces);

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
        rm = trellis::remesh_narrow_band_dc(verts.data(), (int64_t)verts.size()/3,
                                            faces.data(), (int64_t)faces.size()/3, bvh, res, band,
                                            0.0f, signed_remesh);
        printf("  [remesh %.1fs]\n", now()-t); t = now();
        audit("remesh", rm.faces);
        // match the CLI: clean degenerates/unify winding, drop floater components
        if (rm.F() > 0) {
            trellis::clean_mesh(rm.V(), rm.faces);
            audit("clean_mesh", rm.faces);
            int ndrop = trellis::drop_small_components(rm.verts, rm.faces, 0.02f);
            printf("  [clean+drop %.1fs] dropped=%d\n", now()-t, ndrop); t = now();
            audit("drop_components", rm.faces);
        }
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

    if (!do_bake && targets.size() <= 1) {
        // nothing past this point is needed when the run exists only to populate caches
        std::vector<float> dv, dp; std::vector<int32_t> df;
        if (decim > 0) trellis::decimate_cluster(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, {}, decim, dv, df, dp);
        else if (decim == 0) { dv = sverts; df = sfaces; }
        else trellis::decimate_qem(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, targets[0], dv, df);
        printf("  [decimate %.1fs]\n", now()-t);
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
