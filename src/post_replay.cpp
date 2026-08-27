// post-replay — re-run the post-neural stages (weld, hole fill, decimation, UV
// bake, GLB write) from a TRELLIS_DUMP_POST dump, skipping the ~10-minute
// neural pipeline. Development harness for iterating on mesh/texture
// post-processing.
//
//   post-replay <dump.bin> <out.glb> [--box-uv] [--faces N] [--atlas T]
//               [--decim GRID] [--no-weld] [--no-fill]
#include "uv_bake.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "strip_interior.h"
#include "mesh_glb.h"
#include <chrono>
#include <cstdio>
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

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: post-replay <dump.bin> <out.glb> [opts]\n"); return 1; }
    const char* dump = argv[1];
    const char* out = argv[2];
    bool boxuv = false, do_weld = true, do_fill = true, do_bake = true, do_remesh = true, do_snap = true;
    int band = 1;
    bool do_strip = false;
    trellis::StripOpts strip;
    int faces_target = 300000, atlas = 2048, decim = -1;
    const char* save_mesh = nullptr; const char* load_mesh = nullptr;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--box-uv") boxuv = true;
        else if (a == "--faces" && i+1 < argc) faces_target = atoi(argv[++i]);
        else if (a == "--atlas" && i+1 < argc) atlas = atoi(argv[++i]);
        else if (a == "--decim" && i+1 < argc) decim = atoi(argv[++i]);
        else if (a == "--no-weld") do_weld = false;
        else if (a == "--no-fill") do_fill = false;
        else if (a == "--no-bake") do_bake = false;
        else if (a == "--no-remesh") do_remesh = false;
        else if (a == "--band" && i+1 < argc) band = atoi(argv[++i]);
        else if (a == "--no-snap") do_snap = false;
        else if (a == "--strip-interior") do_strip = true;
        else if (a == "--strip-grid" && i+1 < argc) strip.grid = atoi(argv[++i]);
        else if (a == "--strip-depth" && i+1 < argc) strip.depth = atoi(argv[++i]);
        else if (a == "--strip-seal" && i+1 < argc) strip.seal = atoi(argv[++i]);
        else if (a == "--save-mesh" && i+1 < argc) save_mesh = argv[++i];
        else if (a == "--load-mesh" && i+1 < argc) load_mesh = argv[++i];
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
    // Dev aid: cache the post-remesh mesh so the strip can be iterated on
    // without paying weld+fill+bvh+remesh+clean each time (~4 min).
    bool cached = false;
    if (load_mesh) {
        FILE* mf = fopen(load_mesh, "rb");
        if (mf) {
            int mV = 0, mF = 0;
            if (fread(&mV,4,1,mf) == 1 && fread(&mF,4,1,mf) == 1 && mV > 0 && mF > 0) {
                rm.verts.resize((size_t)mV*3);
                rm.faces.resize((size_t)mF*3);
                cached = fread(rm.verts.data(),4,rm.verts.size(),mf) == rm.verts.size()
                      && fread(rm.faces.data(),4,rm.faces.size(),mf) == rm.faces.size();
                if (cached) printf("  [cache] loaded post-remesh V=%d F=%d from %s\n", mV, mF, load_mesh);
            }
            fclose(mf);
            if (!cached) { rm.verts.clear(); rm.faces.clear(); fprintf(stderr, "  [cache] %s unusable, remeshing\n", load_mesh); }
        }
    }
    if (do_remesh && !cached) {
        rm = trellis::remesh_narrow_band_dc(verts.data(), (int64_t)verts.size()/3,
                                            faces.data(), (int64_t)faces.size()/3, bvh, res, band);
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

    if (save_mesh && !cached) {
        FILE* mf = fopen(save_mesh, "wb");
        if (mf) {
            const int mV = (int)(sverts.size()/3), mF = (int)(sfaces.size()/3);
            fwrite(&mV,4,1,mf); fwrite(&mF,4,1,mf);
            fwrite(sverts.data(),4,sverts.size(),mf);
            fwrite(sfaces.data(),4,sfaces.size(),mf);
            fclose(mf);
            printf("  [cache] saved post-remesh mesh V=%d F=%d -> %s\n", mV, mF, save_mesh);
        }
    }

    // Opt-in divergence from the reference: drop the buried sheets of the
    // narrow-band shell before simplifying. The offset surface is two-sided by
    // construction and the input was already two-walled, so most of what QEM is
    // about to spend its budget on can never be seen - and being crumpled, it
    // scores as high-curvature detail and outbids the smooth visible skin.
    if (do_strip) {
        trellis::strip_interior(sverts, sfaces, strip);
        audit("strip_interior", sfaces);
        printf("  [strip %.1fs]\n", now()-t); t = now();
    }

    std::vector<float> dv, dp; std::vector<int32_t> df;
    if (decim > 0) trellis::decimate_cluster(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, {}, decim, dv, df, dp);
    else if (decim == 0) { dv = sverts; df = sfaces; }
    else {
        // match the CLI: faithful QEM port (not the old meshopt/FQMS decimate_simplify)
        trellis::decimate_qem(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, faces_target, dv, df);
        audit("decimate_qem", df);
        trellis::weld_vertices(dv, df, nullptr, 1.0f / ((float)res * 8.0f));
        audit("weld2", df);
        trellis::fill_small_holes(df);
        audit("fill2", df);
        int ndrop2 = trellis::drop_small_components(dv, df, 0.03f);
        if (ndrop2) printf("  dropped %d more comps\n", ndrop2);
        audit("drop2", df);
    }
    printf("  [decimate %.1fs]\n", now()-t); t = now();
    if (!do_bake) { printf("(--no-bake) done\n"); return 0; }

    VoxelPbr vox{&coords, &pbr6, res, do_snap ? &bvh : nullptr};
    const std::vector<float> no_vp;
    trellis::BakedMesh bm = boxuv
        ? trellis::uv_box_project(dv, (int)dv.size()/3, df, (int)df.size()/3, no_vp, atlas, &vox)
        : trellis::uv_bake(dv, (int)dv.size()/3, df, (int)df.size()/3, no_vp, atlas, &vox);
    if (!boxuv && !bm.ok())
        bm = trellis::uv_chart_project(dv, (int)dv.size()/3, df, (int)df.size()/3, no_vp, atlas, &vox);
    printf("  [bake %.1fs]\n", now()-t);
    if (!bm.ok()) { fprintf(stderr, "bake failed\n"); return 1; }
    printf("  [audit] bake: faces in=%zu out=%zu (dropped %lld)\n",
           df.size()/3, bm.faces.size()/3, (long long)(df.size()/3) - (long long)(bm.faces.size()/3));
    trellis::write_glb_textured(out, bm.verts.data(), (int64_t)bm.verts.size()/3, bm.uv.data(),
                                bm.faces.data(), (int64_t)bm.faces.size()/3, bm.base.data(), bm.mr.data(), bm.T,
                                /*double_sided=*/rm.F() == 0);
    printf("wrote %s (atlas %d)\n", out, bm.T);
    return 0;
}
