// ISSUE-001 isolation: at what token count does the HR shape SLAT flow stop
// launching, and which kernel is it?
//
//   trellis-test-token-limit <models_dir> [gpu] [N ...]
//
// Builds the same sparse DiT runner trellis_cli uses for the HR pass
// (shape_flow_1024.gguf, in/out 32 ch, d_cond 1024) over N synthetic voxel
// coordinates and runs ONE forward. No image, no neural stages, no sampler --
// seconds per data point instead of a ~20 minute pipeline run whose token count
// is seed-dependent and cannot be aimed.
//
// The grid-overflow diagnostic in ggml-cuda/common.cuh prints the offending op,
// destination shape and grid dims before CUDA rejects the launch, so a failure
// here names its own cause.
#include "trellis_model.h"
#include "flow_runner.h"
#include "dit.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using std::vector;

// N distinct coordinates in a grid^3 lattice, spread by a stride that is coprime
// with grid^3 so they scatter rather than filling one corner. The DiT only uses
// these for RoPE, so their arrangement matters far less than their count -- but
// clustering them would be an unrepresentative memory pattern.
static vector<std::array<int,3>> make_coords(int n, int grid) {
    vector<std::array<int,3>> c;
    c.reserve(n);
    const int64_t total = (int64_t)grid * grid * grid;
    const int64_t stride = 7919;                      // prime, coprime with any 2^k grid
    for (int i = 0; i < n; ++i) {
        const int64_t idx = (i * stride) % total;
        c.push_back({ (int)(idx / ((int64_t)grid * grid)),
                      (int)((idx / grid) % grid),
                      (int)(idx % grid) });
    }
    return c;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: %s <models_dir> [gpu] [N ...]\n", argv[0]);
        return 1;
    }
    const std::string models = argv[1];
    const int gpu = argc > 2 ? atoi(argv[2]) : 0;

    vector<int> targets;
    for (int i = 3; i < argc; ++i) targets.push_back(atoi(argv[i]));
    if (targets.empty()) targets = { 32768, 60000, 65000, 65535, 65536, 70000, 83110 };

    printf("loading %s/shape_flow_1024.gguf (gpu=%d)\n", models.c_str(), gpu);
    trellis::Model m = trellis::Model::load(models + "/shape_flow_1024.gguf", gpu);

    // Same parameters trellis_cli's HR shape_flow lambda uses.
    trellis::DiTParams p;
    p.in_ch = 32; p.out_ch = 32; p.d_cond = 1024;

    const int Lc = 1370;                   // DINOv3 @1024 conditioning length, order of magnitude
    vector<float> cond((size_t)p.d_cond * Lc, 0.01f);

    for (int n : targets) {
        if (n <= 0) continue;
        printf("\n=== N = %d tokens ===\n", n);
        fflush(stdout);
        const int grid = 128;              // res 2048 / 16; large enough for any N tested
        const vector<std::array<int,3>> coords = make_coords(n, grid);

        trellis::DitRunner* run = trellis::make_sparse_runner(m, p, coords, Lc);
        if (!run) { printf("  runner build FAILED\n"); continue; }

        vector<float> xt((size_t)p.in_ch * n);
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : xt) v = nd(rng);

        // If the launch is rejected, CUDA_CHECK aborts the process here -- which
        // is the answer we came for, printed by the diagnostic above it.
        vector<float> out = run->forward(xt, 1.0f, cond.data());
        double s = 0.0;
        for (float v : out) s += std::fabs((double)v);
        printf("  OK: out %zu values, mean |v| = %.6f\n", out.size(), out.empty() ? 0.0 : s / out.size());
        delete run;
        fflush(stdout);
    }
    m.free();
    printf("\nall targets completed without a rejected launch\n");
    return 0;
}
