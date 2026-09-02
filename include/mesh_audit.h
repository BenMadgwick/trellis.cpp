// Global, ray-based audits over a CLOSED surface: which components enclose
// nothing you can reach, and how much of the delivered surface anyone can see.
//
// Both are integrals over directions rather than local per-face tests. The local
// version of the visibility question was tried and failed (a face cannot tell
// from its own neighbourhood whether it is buried), and the flood-fill version
// of the enclosure question was tried and failed (the input voxel shell leaks:
// a 6-connected exterior flood on the dog's 1.96 M shape voxels encloses five
// voxels, because the paw holes are 22-36 mm wide).
//
// What makes them work here is that they run AFTER the remesh, on a mesh whose
// complement is genuinely partitioned. Jordan-Brouwer applies to a closed
// surface and to nothing else, which is why neither of these belonged on the
// unsigned double cover -- there, the "cavity" between the two sheets IS the
// exterior, so every test answered about the wrong region.
#pragma once
#include <cstdint>
#include <vector>

namespace trellis {

// Delete connected components no ray can escape from: parity bubbles inside a
// solid, and a separate enclosed inner shell. Replaces --strip-interior on the
// Interior path, where it is a topological test rather than a heuristic trough
// in a depth histogram.
//
// It cannot separate an inner wall that is CONNECTED to the outer one, which is
// what a two-walled decode gives (the sheets merge and split rather than meeting
// at a rim). Measured on the bear: one component of 14.09 M faces holding both
// walls, plus one genuinely enclosed 311,928-face shell, which this removed on a
// clean 0-of-512 vs 139-of-512 probe split. That asset still wants
// --strip-interior; Interior mode removes the double cover remesh_dc creates,
// not one the decoder shipped.
//
// The log's "largest component escape rate" reports how much of the surface
// faces inward, but is NOT a decode-artefact detector: the bear reads 27% and
// the jerry can -- a real hollow can whose interior is genuine geometry -- reads
// 24%, while their 10 K visible area is 0.52 against 0.96. Use the tier visible
// fraction to decide whether inward-facing surface is waste.
//
// Per component of >= min_faces faces, `probes` faces are sampled by stride and
// a ray cast along +n and -n from each centroid; the component is EXPOSED if any
// ray leaves without meeting anything. An outer wall escapes with O(1)
// probability per probe, so 2*probes tries make a false "enclosed" verdict on a
// visible surface negligible. Smaller components are kept for
// drop_small_components to judge on size.
//
// A cup's inner surface is kept: it is the same component as its outer surface.
// A pocket reachable only through a narrow neck can be dropped if no probe
// escapes -- acceptable at LOD0 scale, and it is usually connected to the outer
// wall anyway. In place; returns the number of faces removed.
int64_t cull_enclosed_components(std::vector<float>& verts, std::vector<int32_t>& faces,
                                 int probes = 256, int min_faces = 100, bool verbose = true);

struct VisibleAudit {
    int64_t faces = 0, faces_hit = 0;
    int64_t rays_cast = 0, rays_hit = 0;
    double area = 0.0, area_hit = 0.0;
    double face_frac() const { return faces ? (double)faces_hit / (double)faces : 0.0; }
    double area_frac() const { return area > 0.0 ? area_hit / area : 0.0; }
    // Whether there were enough rays for the fraction to mean anything.
    //
    // A ray marks at most one face, so faces_hit can never exceed rays_hit --
    // on a mesh with millions of faces the metric saturates on the RAY budget
    // and reports rays/faces rather than visibility. Measured on the dog: the
    // unsigned 7.75 M-face high-poly and the Interior 3.97 M-face one returned
    // 1,113,885 and 1,113,122 hit faces respectively -- the same surface, seen
    // by the same rays -- yet scored 0.14 and 0.28 purely on denominator.
    //
    // Below ~4 hits per visible face the number is a LOWER BOUND, not an
    // occlusion measurement. It is still comparable between two meshes audited
    // at the same settings; it is not comparable to an absolute threshold. The
    // decimated tiers, where a face budget of thousands meets millions of rays,
    // are where an absolute reading is meaningful -- and they are also the only
    // place the answer changes what ships.
    bool undersampled() const { return faces_hit && rays_hit < 4 * faces_hit; }
};

// Fraction of the surface that is a FIRST hit from outside the bounding sphere:
// `ndirs` Fibonacci directions x `grid`^2 orthographic rays each, marking the
// face each ray lands on. This is the triangle-budget efficiency number -- on a
// 10 K tier decimated from the unsigned double cover it reads ~0.6, because 40%
// of the budget went to walls that are never a first hit.
//
// Check undersampled() before comparing the result to a threshold.
VisibleAudit visible_fraction(const std::vector<float>& verts, const std::vector<int32_t>& faces,
                              int ndirs = 512, int grid = 128);

}  // namespace trellis
