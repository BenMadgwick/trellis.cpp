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

// `coords` are voxel indices in a `grid`^3 lattice. Two callers:
//
//   structure stage  ss_coords output at grid 32, no colour -- available ~30 s
//                    in, catches a wrong object or wrong proportions
//   textured stage   the PBR voxel field at res 512/1024 with per-voxel base
//                    RGB -- available after the texture decode and BEFORE the
//                    expensive mesh post-processing, so it catches an asset
//                    that is the right shape but looks wrong
//
// `rgb`, when given, is [N*3] base colour in 0..1 parallel to `coords`.
// Fields finer than `max_grid` are binned down to it (colour averaged), which is
// what keeps a 7 M-voxel textured preview to seconds.
//
// Output is a 4x2 turntable, so 4*tile x 2*tile. Positions are rotated into the
// glTF frame (mesh_glb.cpp's (x,y,z)->(x,z,-y)) so the preview stands up the same
// way the exported asset will. Returns false if nothing could be written.
bool write_vox_preview(const std::vector<std::array<int,3>>& coords, int grid,
                       const std::string& path, int tile = 320,
                       const std::vector<float>* rgb = nullptr, int max_grid = 160);

}  // namespace trellis
