// The master: everything the neural stages produce, before any BUDGET decision.
//
// The decoded mesh plus the sparse PBR volume the bake samples. Triangle target,
// atlas size and texture format are all chosen downstream of it, so one master
// serves any number of face targets -- post-replay consumes it and re-decimates
// without re-running a neural stage.
//
// WHY THIS IS NOT THE TRELLIS_DUMP_POST LAYOUT
//
// That layout carries ONE resolution, and two different resolutions are in play:
//
//   mesh_res  the structure grid the mesh was decoded and contoured at
//   pbr_res   the PBR volume's own resolution, which the texture stage drops to
//             512 on a dense res-1024 asset (a clean coarse PBR bakes onto the
//             res-1024 mesh without the partial-coverage speckle)
//
// A consumer handed the single field uses it for both. Four of its five uses --
// the weld epsilon, the remesh grid, the band auto-scale and the normal-bake
// epsilon -- want mesh_res; only the PBR sampler wants pbr_res. So on exactly
// the dense assets a cache is most valuable for, a replay contoured at grid 512
// instead of 1024 and produced a coarser shell than the run it was replaying,
// silently.
//
// The legacy layout is still READ (headerless, single res) so existing dumps
// keep working; it is no longer written for masters. --dump-post keeps writing
// it, because an external consumer depends on that exact byte layout.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trellis {

struct Master {
    int mesh_res = 0;   // structure/remesh grid -- weld eps, remesh, band, normal eps
    int pbr_res  = 0;   // PBR volume resolution -- the texel sampler, and nothing else
    std::vector<float>              verts;   // [V*3]
    std::vector<int32_t>            faces;   // [F*3]
    std::vector<std::array<int,3>>  coords;  // [Mv]  sparse PBR voxel coordinates
    std::vector<float>              pbr6;    // [Mv*6] base_color.rgb, metallic, roughness, alpha

    // True when loaded from the headerless single-resolution layout, in which
    // case mesh_res == pbr_res because the file could not say otherwise.
    bool legacy = false;

    int V() const { return (int)(verts.size() / 3); }
    int F() const { return (int)(faces.size() / 3); }
    int Mv() const { return (int)coords.size(); }
};

// Written to <path>.part and renamed, so a run killed mid-write leaves no
// truncated file for a later run to read back as a complete cache entry.
bool save_master(const std::string& path, const Master& m);

// Accepts both the current layout and the legacy headerless one; sets
// m.legacy on the latter.
bool load_master(const std::string& path, Master& m);

}  // namespace trellis
