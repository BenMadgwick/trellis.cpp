#include "trellis_args.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace trellis {

void print_usage(const char* argv0, bool server) {
    if (server) {
        fprintf(stderr,
            "usage: %s [--host H] [--port P] [--models DIR] [--gpu N] [generation defaults...]\n",
            argv0);
    } else {
        fprintf(stderr,
            "usage: %s <image.png> <out.glb> [options]\n"
            "   or: %s --image <image.png> --output <out.glb> [options]\n",
            argv0, argv0);
    }
    fprintf(stderr,
        "\n"
        "  -i, --image PATH        input image                  (image->3D)\n"
        "  -o, --output PATH       output .glb                  (default model.glb)\n"
        "      --copyright TEXT    glTF asset.copyright metadata\n"
        "  -m, --models DIR        GGUF model directory\n"
        "      --gpu N             GPU index, <0 = CPU          (default 0)\n"
        "  -s, --seed N            RNG seed                     (default 42)\n"
        "      --res 512|1024|1536 geometry resolution\n"
        "      --max-tokens N      HR token budget              (default 49152)\n"
        "      --bg-removal MODE   threshold | birefnet   (default: auto -- a pre-matted\n"
        "                          image keeps its alpha; otherwise BiRefNet when its model\n"
        "                          is present. The plain threshold matte cuts out specular\n"
        "                          highlights, which the flow then turns into holes.)\n"
        "      --birefnet          alias for --bg-removal birefnet\n"
        "      --no-texture        geometry only\n"
        "      --xatlas            xatlas UV unwrap (default)\n"
        "      --box-uv            voxel-native box projection (faster)\n"
        "      --faces N           QEM face budget (default: 300K @1024 / 150K @512)\n"
        "      --normal-map        bake a tangent-space normal map from the high-poly\n"
        "                          (needs a single-cover source: any --remesh-mode but\n"
        "                          'unsigned', or 'unsigned' plus --strip-interior)\n"
        "      --normal-search K   normal-bake search bound, in local edge lengths (default 2)\n"
        "      --normal-cap-shells K  cap the normal-bake search at K offset-shell\n"
        "                          thicknesses (default 4, 0 = uncapped) -- stops the\n"
        "                          search tunnelling out the far side of the object\n"
        "      --normal-min-dot K  abandon the normal bake below this winding consensus\n"
        "                          (default 0.85, 0 disables) -- catches an asset whose\n"
        "                          buried sheets the strip failed to remove\n"
        "      --remesh-mode M     which field the remesh contours: auto (default),\n"
        "                          interior, signed5, unsigned. 'interior' unions the\n"
        "                          eps band with the ray-parity interior, so the buried\n"
        "                          wall is never built and the budget is not split; the\n"
        "                          outer surface is unchanged. 'auto' adds a fallback to\n"
        "                          unsigned+strip if the output audit says it failed.\n"
        "      --sign-rays N       max parity directions per grid vertex (default 64; 8\n"
        "                          cube diagonals are cast first and decide it when they\n"
        "                          agree, which is nearly everywhere)\n"
        "      --no-cull           keep components no ray can escape from (default: drop\n"
        "                          them -- a two-walled decode's inner bag, parity bubbles)\n"
        "      --fill-hipoly P     fan-fill high-poly boundary loops up to perimeter P\n"
        "                          (default 0.25, 0 = off) so QEM and the normal bake\n"
        "                          never see a hole\n"
        "      --remesh-project X  lerp each dual vertex X of the way onto the input\n"
        "                          surface (default 0; 0.9 is the reference's own value)\n"
        "      --strip-interior    drop the buried sheets of the narrow-band shell before\n"
        "                          simplifying, so the budget is spent on the visible\n"
        "                          surface (no-op when nothing is buried). Only needed on\n"
        "                          --remesh-mode unsigned; the others have no buried sheet\n"
        "      --band N            narrow-band DC remesh band width (default: auto —\n"
        "                          res/512, i.e. 1 @512 / 2 @1024, which suppresses the\n"
        "                          res-1024 outer-skin speckle; N forces that width)\n"
        "      --decim GRID        legacy cluster-grid decimation (default: quadric\n"
        "                          simplify to 300K faces @1024 / 150K @512; 0 = none)\n"
        "      --atlas PX          UV atlas size (default 2048 @1024 / 1024 @512)\n"
        "      --tex-res N         texture PBR resolution 512/1024 (default: auto — drops\n"
        "                          a dense res-1024 decode to a clean res-512 PBR volume)\n"
        "      --webp on|off       encode GLB textures as WebP (default: on when built with\n"
        "                          WebP support; off = PNG)\n"
        "    two-stage generation (structure preview, then resume):\n"
        "      --vox-only          stop after the sparse-structure decode (~1-2 s) instead\n"
        "                          of running shape + texture + bake (~45-60 s)\n"
        "      --vox-render PNG    write a 4-view voxel preview (front / three-quarter /\n"
        "                          side / top) so a bad structure can be spotted and the\n"
        "                          job abandoned before it costs the GPU time\n"
        "      --save-vox FILE     write the resume cache (voxels + DINOv3 conditioning)\n"
        "      --load-vox FILE     resume from that cache, skipping preprocess, DINOv3 and\n"
        "                          the structure flow. Needs NO input image: the cache holds\n"
        "                          the conditioning, which is all the later stages ever used\n"
        "                          the image for\n"
        "      --dump-bg           also write the background-removal cutout as <out>_cutout.png\n"
        "      --bg-only           background removal only: write the cutout and skip the rest\n"
        "      --f32               f32 sparse-conv compute\n"
        "      --no-fa             disable FlashAttention\n"
        "      --require-gpu       refuse CPU fallback\n"
        "      --threads N         CPU backend threads      (default all cores)\n"
        "      --gss F  --gsh F    guidance strengths\n"
        "      --host H  --port P  trellis-server bind address\n"
        "      --voxply            also dump the voxel point cloud as .ply\n"
        "      --dump-slat         dump the structured latent to disk\n"
        "  -h, --help              show this help\n");
}

bool parse_args(int argc, char** argv, TrellisParams& p) {
    int positional = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "[trellis] %s needs a value\n", name); return nullptr; }
            return argv[++i];
        };
        auto need = [&](const char* name) -> const char* {
            const char* v = next(name);
            return v;
        };

        if      (a == "-h" || a == "--help")    { p.help = true; return false; }
        else if (a == "-i" || a == "--image")   { const char* v = need(a.c_str()); if (!v) return false; p.image = v; }
        else if (a == "-o" || a == "--output")  { const char* v = need(a.c_str()); if (!v) return false; p.output = v; }
        else if (a == "--copyright")            { const char* v = need(a.c_str()); if (!v) return false; p.copyright = v; }
        else if (a == "-m" || a == "--models")  { const char* v = need(a.c_str()); if (!v) return false; p.models = v; }
        else if (a == "--gpu")                  { const char* v = need(a.c_str()); if (!v) return false; p.gpu = atoi(v); }
        else if (a == "-s" || a == "--seed")    { const char* v = need(a.c_str()); if (!v) return false; p.seed = (uint32_t)atoi(v); }
        else if (a == "--res")                  { const char* v = need(a.c_str()); if (!v) return false; p.set_res(atoi(v)); }
        else if (a == "--max-tokens")           { const char* v = need(a.c_str()); if (!v) return false; p.max_tokens = atoi(v); }
        else if (a == "--bg-removal")           { const char* v = need(a.c_str()); if (!v) return false; p.birefnet = (std::strcmp(v, "birefnet") == 0) ? 1 : 0; }
        else if (a == "--birefnet")             { p.birefnet = 1; }
        else if (a == "--no-texture")           { p.texture = false; }
        else if (a == "--xatlas")               { p.xatlas = true; }
        else if (a == "--box-uv")               { p.xatlas = false; }
        else if (a == "--faces")                { const char* v = need(a.c_str()); if (!v) return false; p.faces = atoi(v); }
        else if (a == "--strip-interior")       { p.strip_interior = true; }
        else if (a == "--remesh-mode")          { const char* v = need(a.c_str()); if (!v) return false;
                                                  const std::string m = v;
                                                  if (m != "auto" && m != "interior" && m != "signed5" && m != "unsigned") {
                                                      fprintf(stderr, "--remesh-mode: expected auto|interior|signed5|unsigned, got '%s'\n", v);
                                                      return false;
                                                  }
                                                  p.remesh_mode = m; }
        else if (a == "--sign-rays")            { const char* v = need(a.c_str()); if (!v) return false; p.sign_rays = atoi(v); }
        else if (a == "--no-cull")              { p.cull = false; }
        else if (a == "--fill-hipoly")          { const char* v = need(a.c_str()); if (!v) return false; p.fill_hipoly = (float)atof(v); }
        else if (a == "--remesh-project")       { const char* v = need(a.c_str()); if (!v) return false; p.remesh_project = (float)atof(v); }
        else if (a == "--normal-map")           { p.normal_map = true; }
        else if (a == "--normal-search")        { const char* v = need(a.c_str()); if (!v) return false; p.normal_search = (float)atof(v); }
        else if (a == "--normal-min-dot")       { const char* v = need(a.c_str()); if (!v) return false; p.normal_min_dot = (float)atof(v); }
        else if (a == "--normal-cap-shells")    { const char* v = need(a.c_str()); if (!v) return false; p.normal_cap_shells = (float)atof(v); }
        else if (a == "--band")                 { const char* v = need(a.c_str()); if (!v) return false; p.band = atoi(v); }
        else if (a == "--decim")                { const char* v = need(a.c_str()); if (!v) return false; p.decim = atoi(v); }
        else if (a == "--atlas" || a == "--tex"){ const char* v = need(a.c_str()); if (!v) return false; p.tex = atoi(v); }
        else if (a == "--tex-res")              { const char* v = need(a.c_str()); if (!v) return false; p.tex_res = atoi(v); }
        else if (a == "--webp")                 { const char* v = need(a.c_str()); if (!v) return false;
                                                  p.webp = (std::strcmp(v,"off")==0 || std::strcmp(v,"0")==0 || std::strcmp(v,"false")==0) ? 0
                                                         : (std::strcmp(v,"on")==0 || std::strcmp(v,"1")==0 || std::strcmp(v,"true")==0) ? 1 : -1; }
        else if (a == "--vox-only")             { p.vox_only = true; }
        else if (a == "--vox-render")           { const char* v = need(a.c_str()); if (!v) return false; p.vox_render = v; }
        else if (a == "--save-vox")             { const char* v = need(a.c_str()); if (!v) return false; p.save_vox = v; }
        else if (a == "--load-vox")             { const char* v = need(a.c_str()); if (!v) return false; p.load_vox = v; }
        else if (a == "--dump-bg")              { p.dump_bg = true; }
        else if (a == "--bg-only")              { p.bg_only = true; p.dump_bg = true; }
        else if (a == "--f32")                  { p.f32 = true; }
        else if (a == "--no-fa")                { p.no_fa = true; }
        else if (a == "--require-gpu")          { p.require_gpu = true; }
        else if (a == "--threads")              { const char* v = need(a.c_str()); if (!v) return false; p.threads = atoi(v); }
        else if (a == "--gss")                  { const char* v = need(a.c_str()); if (!v) return false; p.gss = (float)atof(v); }
        else if (a == "--gsh")                  { const char* v = need(a.c_str()); if (!v) return false; p.gsh = (float)atof(v); }
        else if (a == "--host")                 { const char* v = need(a.c_str()); if (!v) return false; p.host = v; }
        else if (a == "--port")                 { const char* v = need(a.c_str()); if (!v) return false; p.port = atoi(v); }
        else if (a == "--voxply")               { p.voxply = true; }
        else if (a == "--dump-slat")            { p.dump_slat = true; }
        else if (!a.empty() && a[0] == '-')     { fprintf(stderr, "[trellis] unknown option: %s\n", a.c_str()); return false; }
        else if (positional == 0)               { p.image  = a; positional = 1; }
        else if (positional == 1)               { p.output = a; positional = 2; }
        else                                    { fprintf(stderr, "[trellis] unexpected argument: %s\n", a.c_str()); return false; }
    }
    // A resumed run takes no image, so its ONE bare positional is the output.
    // Without this, `trellis-cli --load-vox c.vox out.glb` -- the obvious way to
    // write it -- puts out.glb in `image`, and the model is silently written to
    // the default model.glb instead.
    if (!p.load_vox.empty() && positional == 1) {
        p.output = p.image;
        p.image.clear();
    }
    return true;
}

}  // namespace trellis
