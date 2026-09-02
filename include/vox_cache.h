// Resume cache for the two-stage pipeline: everything downstream of the sparse
// structure decode, and nothing else.
//
// Stages [1] preprocess and [2] DINOv3 exist only to produce the conditioning;
// after that the source image is never read again (trellis_cli.cpp uses `chw`
// solely at the dinov3_encode calls). Stage [3] produces the voxel coords. So
// coords + conditioning IS the complete resume state, and a resumed run needs
// no image, no BiRefNet and no DINOv3 -- which is why --load-vox takes no
// --image, unlike the two-stage sketch in the issue.
//
// The negative conditioning is not stored: it is a zero vector of the same
// length, reconstructed on load.
#pragma once
#include <array>
#include <string>
#include <vector>

namespace trellis {

struct VoxCache {
    bool  cascade = false;
    int   hr_res  = 1024;
    int   grid    = 32;          // voxel lattice of `coords`
    uint32_t seed = 0;           // the seed that produced this structure
    std::vector<std::array<int,3>> coords;
    std::vector<float> cond;     // [1024 * Lc]   DINOv3 @512
    std::vector<float> cond1024; // [1024 * Lc1024] DINOv3 @1024 (cascade only)
    int Lc() const     { return (int)(cond.size() / 1024); }
    int Lc1024() const { return (int)(cond1024.size() / 1024); }
};

bool save_vox_cache(const std::string& path, const VoxCache& v);
bool load_vox_cache(const std::string& path, VoxCache& v);

}  // namespace trellis
