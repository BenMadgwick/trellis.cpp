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

    // The LR shape flow's output, and the res-512 coordinates it upsamples to.
    // Optional: absent in a cache written before the HR pass ran.
    //
    // Worth carrying because the LR pass is cheap (~30 s) but sets the COARSE
    // STRUCTURE, and the expensive HR and texture passes are what a low-step
    // preview wants to skimp on. Sharing it makes the preview a faithful
    // predictor of the final asset rather than merely a cheaper one -- both runs
    // then start the HR flow from identical coarse geometry.
    //
    // lr_steps records what it was sampled with, so a full run will NOT silently
    // inherit a 4-step structure from a preview: if the cached count is lower
    // than the run wants, the LR pass is recomputed.
    int lr_steps = 0;                        // 0 = no LR data in this cache
    std::vector<float> lr_slat;              // [32 * coords.size()] denormalised
    std::vector<std::array<int,3>> hr_coords;// res-512 upsampled coordinates
    int Lc() const     { return (int)(cond.size() / 1024); }
    int Lc1024() const { return (int)(cond1024.size() / 1024); }
};

bool save_vox_cache(const std::string& path, const VoxCache& v);
bool load_vox_cache(const std::string& path, VoxCache& v);

}  // namespace trellis
