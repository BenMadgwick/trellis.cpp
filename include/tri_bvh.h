// Median-split AABB tree over triangles with exact closest-point queries.
// Non-owning: the caller's vertex/index arrays must outlive the tree. Used by
// the texture bake (per-texel snap to the original surface, mirroring the
// reference's cuBVH unsigned_distance) and by the narrow-band remesh (UDF
// evaluation).
#pragma once
#include <cstdint>
#include <vector>

namespace trellis {

class TriBvh {
public:
    struct Hit {
        float dist2 = 1e30f;
        int32_t face = -1;
        float point[3] = {0, 0, 0};
    };

    // First surface a ray meets. Used to decide which sheet of the narrow-band
    // shell is the one you can actually see (see strip_interior.h).
    struct RayHit {
        float t = 1e30f;      // distance along dir, which need not be normalised
        int32_t face = -1;
    };

    static TriBvh build(const float* verts, int64_t V, const int32_t* faces, int64_t F);

    Hit closest(const float p[3], float max_dist = 1e30f) const;
    RayHit ray(const float org[3], const float dir[3], float max_t = 1e30f) const;

    // How many triangles the ray meets in (1e-7, max_t]. ONE traversal: unlike
    // ray(), nothing is pruned by a nearest hit, so every node whose slab
    // interval overlaps the ray is visited and every triangle in it tested.
    //
    // This is the parity primitive. The old parity_sign() answered the same
    // question by calling ray() in a restart loop -- one full traversal per
    // crossing, plus an epsilon nudge past each hit that is simultaneously too
    // large for a thin wall and too small for a coincident pair. Counting in a
    // single descent is both faster and exact.
    //
    // Two-sided, like ray(): parity counts crossings and must not care which way
    // a triangle faces. That is the whole point -- the decoder mesh's winding is
    // a fixed-table patchwork with no global orientation (~16-20% of shared
    // edges disagree with their neighbour), so any method that reads a sign off
    // the input is answering noise.
    int count_hits(const float org[3], const float dir[3], float max_t) const;

    bool empty() const { return nodes_.empty(); }

private:
    struct Node {
        float bmin[3], bmax[3];
        int32_t left;    // internal: index of left child (right = left+1); leaf: first prim index
        int32_t count;   // 0 for internal nodes; >0 = leaf primitive count
    };
    std::vector<Node> nodes_;
    std::vector<int32_t> prim_;
    const float* verts_ = nullptr;
    const int32_t* faces_ = nullptr;
};

}  // namespace trellis
