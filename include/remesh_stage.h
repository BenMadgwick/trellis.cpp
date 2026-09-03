// Steps 3-6 of the mesh stage, in one place: narrow-band DC remesh, clean,
// drop floaters, fan-fill high-poly holes, cull enclosed components -- plus the
// `auto` gate that judges Interior mode on its OUTPUT and falls back to
// unsigned + strip.
//
// This existed twice: once in trellis_cli.cpp and once in post_replay.cpp, with
// the same thresholds reached through different code (open_per_1k over an
// unordered_map on one side, a boundary count from post-replay's audit() on the
// other). Two copies of the gate is one copy too many -- post_replay's fallback
// did not update its remesh mode afterwards, so later single-cover decisions
// there could still read the pre-fallback value. A stage whose whole job is to
// decide something must decide it once.
#pragma once
#include "remesh_dc.h"
#include "tri_bvh.h"
#include "uv_bake.h"   // clean_mesh, drop_small_components, fill_holes
#include <cstdint>
#include <functional>
#include <vector>

namespace trellis {

// Boundary edges per 1000 faces over the welded index space. The `auto` gate
// reads this on the finished high-poly rather than predicting it from the
// input, because Interior's only failure modes show up in the output.
double open_per_1k(const std::vector<int32_t>& faces);

struct RemeshStageOpts {
    int        res           = 1024;  // contouring grid
    int        band          = 0;     // 0 => auto: max(1, res/512), so the
                                      // world-space offset is resolution-independent
    float      project       = 0.0f;
    RemeshMode mode          = RemeshMode::Interior;
    bool       mode_auto     = true;  // let a failed output audit fall back
    int        sign_rays     = 64;
    int        parity_coarse = 2;
    bool       cull          = true;
    float      fill_hipoly   = 0.25f;
    const char* sign_dump    = nullptr;  // parity dump (post-replay experiments)
    bool       verbose       = false;    // per-stage timings

    // Optional per-stage hook, called after each step with the current faces.
    // post-replay audits boundary and non-manifold counts between the steps;
    // folding the steps into one function would otherwise swallow exactly the
    // diagnostics that harness exists to print.
    std::function<void(const char* tag, const std::vector<int32_t>& faces)> on_stage;
};

struct RemeshStageResult {
    Mesh       mesh;                              // empty if the remesh produced nothing
    RemeshMode final_mode  = RemeshMode::Interior;// what was actually built
    bool       fell_back   = false;               // the auto gate rejected `mode`
    bool       want_strip  = false;               // caller must strip (unsigned path)
    double     open1k      = 0.0;                 // of the mesh returned
};

// `bvh` must cover the SAME verts/faces passed here (the pre-remesh, welded,
// hole-filled mesh): the remesh's UDF queries it, and the caller's albedo snap
// wants the same tree afterwards.
//
// `want_strip_in` carries the caller's --strip-interior request; the result's
// want_strip is that OR'd with a fallback having fired.
RemeshStageResult remesh_stage(const std::vector<float>& verts,
                               const std::vector<int32_t>& faces,
                               const TriBvh& bvh,
                               const RemeshStageOpts& o,
                               bool want_strip_in = false);

}  // namespace trellis
