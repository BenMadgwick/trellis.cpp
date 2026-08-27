// Drop every sheet of the mesh except the one you can actually see.
//
// remesh_narrow_band_dc contours f(x) = UDF(x) - eps. The field is UNSIGNED, so
// the eps level set is an offset surface on BOTH sides of the input: the output
// is the boundary of a slab, not a skin. The FlexiDualGrid mesh it is handed is
// itself already two-walled (the decoded voxel band has thickness), so the
// remesh emits four nested sheets, of which exactly one is the object.
//
// That costs twice over. The buried sheets are invisible, and because offsetting
// a concave surface inward produces cusps and self-intersections, they carry far
// higher curvature than the smooth skin - so a quadric decimator scores them as
// detail worth keeping and spends almost the whole face budget on them.
//
// Measured on a 297,734-face bear: 8,710 faces were visible (2.9%). Deleting the
// rest changed nothing a camera can see - 20,000 probe rays hit the identical
// point, to 0.000mm at the 99th percentile.
//
// This is opt-in (--strip-interior) because it is a deliberate divergence from
// the reference pipeline.
#pragma once
#include <cstdint>
#include <vector>

namespace trellis {

struct StripOpts {
    int grid = 1024;    // voxel grid resolution for the exterior flood fill
    int seal = 1;       // dilations of the shell before flooding, to close leaks
    int depth = -1;     // cells from the exterior to keep; -1 = find the trough
    int threads = 0;    // reserved
    bool verbose = true;
};

// Rewrites verts/faces in place, keeping only the faces that bound the
// exterior. Returns the number of faces removed.
int64_t strip_interior(std::vector<float>& verts, std::vector<int32_t>& faces,
                       const StripOpts& opt = StripOpts());

}  // namespace trellis
