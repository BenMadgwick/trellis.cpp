// Narrow-band UDF dual-contouring remesh, ported from the reference
// (CuMesh remeshing.py::remesh_narrow_band_dc + simple_dual_contour.cu; see
// docs/spec/27-reference-postprocess.md §4). Rebuilds a noisy voxel-derived
// mesh as the closed, consistently-oriented offset surface at distance
// eps = band·scale/res around it — one watertight manifold that downstream
// simplification and UV unwrapping can digest.
#pragma once
#include "dual_grid.h"

namespace trellis {

class TriBvh;

// `bvh` must be built over (verts, faces). Returns an empty mesh on failure
// (caller keeps the un-remeshed path).
// `project_back` lerps each dual vertex toward its closest point on the input
// surface (o_voxel postprocess `remesh_project`; 0 disables).
//
// Default 0 to match the HF space — the config that produced the reference GLBs
// users compare against — which passes remesh_project=0 (trellis2-space/app.py).
// o_voxel's own to_glb default is 0.9 and TRELLIS.2's app.py inherits it, so the
// knob is kept: measured on the issue #1 goblin, 0.9 improves the surface
// (dihedral 8.08->7.41deg, >60deg spikes 1.12%->0.64%) but quadruples our UV
// merge clusters (2519->10248), so it is NOT a free win in this pipeline.
// Which scalar field gets contoured. The band, the UDF, the quad tables and the
// winding rule are identical in all three; only f(v) differs.
//
//   Unsigned   f = UDF - eps           the reference. A double cover: an
//                                      unsigned field has no inside, so the
//                                      eps level set wraps BOTH sides of the
//                                      input and half of every face budget
//                                      lands on a wall nobody can see. The two
//                                      walls cannot be separated afterwards --
//                                      measured, they merge and split
//                                      throughout rather than meeting at a rim
//                                      (see strip_interior.cpp).
//
//   Signed5    f = parity5 * UDF       one surface, but a PRODUCT: a wrong sign
//                                      at one vertex moves the zero set, so the
//                                      surface must pass between that vertex
//                                      and its neighbours and a hole opens.
//                                      Measured 16.9 open edges per 1000 faces
//                                      on the jerry can. It also deletes thin
//                                      sheets, which have no interior to sign.
//                                      Kept for A/B only.
//
//   Interior   f = min(UDF - eps, 2*eps*(0.5 - F))       what we ship.
//
// Interior contours the boundary of {UDF < eps} UNION {parity interior}, with F
// the fraction of directions along which the input is crossed an odd number of
// times. Three properties follow structurally rather than by tuning:
//
//   - The outer sheet is UNCHANGED, bit for bit. The parity term only binds
//     where UDF > 2*eps, and UDF is 1-Lipschitz, so on any edge that crosses
//     the outer sheet the far corner has UDF <= eps + cell <= 2*eps for every
//     band >= 1. Same corners, same crossings, same dual vertices.
//   - The inner sheet is gone: deep inside a solid the parity term is -eps, so
//     f < 0 where the unsigned field was positive and no crossing exists.
//   - A union cannot tear. Inside the band f < 0 whatever F says, so a sign
//     error near the surface has no authority at all; outside it, one makes a
//     closed bump or a closed bubble, which drop_small_components and the
//     component cull already remove. That is the difference from Signed5.
//
// Thin sheets fall out of the same expression with no thickness threshold: a
// 1 mm tabletop has no parity interior, so F <= 0.5 everywhere, the second term
// is always +eps, and Interior reduces EXACTLY to Unsigned there.
//
// Parity, not a signed distance: the input's winding is a fixed-table patchwork
// (~16-20% of shared edges disagree with their neighbour, exact generalized
// winding numbers ambiguous on 62-73% of samples 1.5 mm off the surface), so an
// angle-weighted pseudonormal reads noise. Counting crossings does not care.
enum class RemeshMode { Unsigned, Signed5, Interior };

// `sign_rays` caps the directions per parity query in Interior mode (8 cube
// diagonals are cast first and decide it outright when unanimous). `sign_dump`,
// when set, writes float32 x,y,z,udf,F per evaluated band vertex for the R1
// histogram; nullptr disables it.
//
// `coarse` is the coarse-to-fine stride for the parity pass: F is evaluated on
// every `coarse`-th grid vertex first, and a fine vertex inherits for free when
// all eight surrounding coarse samples agree. F is 0 or 1 across almost the
// whole volume, so most of them do. 1 disables it (evaluate every vertex).
Mesh remesh_narrow_band_dc(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                           const TriBvh& bvh, int res, int band = 1,
                           float project_back = 0.0f,
                           RemeshMode mode = RemeshMode::Unsigned,
                           int sign_rays = 64, const char* sign_dump = nullptr,
                           int coarse = 2);

}  // namespace trellis
