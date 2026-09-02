// Fast CPU preview render of the sparse-structure stage's voxel output.
//
// The sparse structure decode finishes in ~1-2 s, long before the ~45-60 s of
// shape flow, texture flow and PBR bake that follow. If the structure is wrong
// -- bad proportions, a hallucinated second object, a subject fused to its
// backdrop -- it is obvious in the voxels, and every second spent after that
// point is wasted. On a shared GPU that time is also somebody else's queue.
//
// So this renders the voxels directly: no mesh, no textures, no GPU. Four fixed
// views (front, three-quarter, side, top) rasterised into one 2x2 composite so a
// single thumbnail answers "is this the right object, the right way up, roughly
// the right shape?".
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trellis {

// `coords` are voxel indices in a `grid`^3 lattice (ss_coords output, grid 32).
// Writes a PNG of 2*tile x 2*tile. Returns false if the file could not be
// written or there is nothing to draw.
bool write_vox_preview(const std::vector<std::array<int,3>>& coords, int grid,
                       const std::string& path, int tile = 320);

}  // namespace trellis
