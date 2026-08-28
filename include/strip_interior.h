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

class TriBvh;

struct StripOpts {
    int grid = 1024;    // voxel grid resolution for the exterior flood fill
    int seal = 1;       // dilations of the shell before flooding, to close leaks
    int depth = -1;     // cells from the exterior to keep; -1 = find the trough
    int threads = 0;    // reserved
    bool verbose = true;
    // Classify each face by which side of it is open, instead of thresholding a
    // depth histogram.
    //
    // The histogram cannot see a two-sheet shell: its walls are 2*eps ~ 3.9 mm
    // apart, four cells at 1024^3, and the seal dilation that stops the exterior
    // flood leaking also fills that gap -- the detector must destroy the signal
    // before it can read it. Worse, per-face depth is taken from the shallowest
    // cell in the triangle's bounding box, so a large or diagonal triangle picks
    // up cells nowhere near itself, and any threshold then cuts through the
    // middle of the outer wall (measured: 821,183 boundary edges on a dog).
    //
    // The remesh is closed and consistently wound, so an outer-wall face's
    // normal points into the exterior while an inner-wall face's points into the
    // enclosed cavity. Probing a short way along the normal separates them
    // exactly, per face, with no threshold and no histogram.
    bool outward = false;
    bool shrink = true; // project onto the input surface before finding the seam
    float fold_deg = 120.f; // dihedral above which an edge is a fold, post-shrink
    int  smooth = 8;    // diffusion passes over face adjacency before thresholding
                        //   the side-of-input sign; 0 uses the raw sign
    // The PRE-remesh mesh and a BVH over it, for the outward test below.
    // remesh_dc already builds exactly this to evaluate its UDF, so passing it
    // here costs nothing.
    const float*   ref_verts = nullptr;
    const int32_t* ref_faces = nullptr;
    int64_t        ref_V = 0, ref_F = 0;
    const TriBvh*  ref_bvh = nullptr;
};

// Rewrites verts/faces in place, keeping only the faces that bound the
// exterior. Returns the number of faces removed.
int64_t strip_interior(std::vector<float>& verts, std::vector<int32_t>& faces,
                       const StripOpts& opt = StripOpts());

}  // namespace trellis
