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
// `signed_field` contours the SIGNED zero level set instead of the unsigned
// eps-offset, giving ONE surface rather than a double cover.
//
// The unsigned field has no inside, so its level set wraps both sides of the
// input: half of every face budget then lands on a wall nobody can see, and the
// two walls cannot be separated afterwards -- measured, they merge and split
// throughout rather than meeting at a rim (see strip_interior.cpp). The fix is
// to never build the double cover.
//
// Requires an input that encloses a volume with consistent winding, which the
// decoded mesh does: measured signed volume / area of 21.1 mm on the dog, 25.2
// on the table, 4.2 on the jerry can -- body-scale, not eps-scale, and cleanly
// signed rather than near zero. The teddy bear is the exception at 1.78 mm; its
// decode is already a thin two-walled bag, so it has no interior to sign and
// must keep the unsigned path.
//
// Unlike MeshUDF / DualMesh-UDF / NSDUDF, which must ESTIMATE a pseudo-sign from
// the gradients of a neural field, we have the mesh itself and can take a real
// sign from an angle-weighted pseudonormal at the closest point.
Mesh remesh_narrow_band_dc(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                           const TriBvh& bvh, int res, int band = 1,
                           float project_back = 0.0f, bool signed_field = false);

}  // namespace trellis
