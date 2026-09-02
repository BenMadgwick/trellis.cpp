// GPU parity: the Interior remesh's dominant cost, moved to the hardware built
// for it.
//
// Every grid vertex is independent and every ray within one is independent, so
// the pass is embarrassingly parallel -- and BVH ray traversal is precisely what
// a GPU is for. Measured on the CPU: the bear spent 277 M ray casts and 472 s in
// this pass, and a 34.6 M-face pallet ran over an hour without finishing.
//
// This mirrors remesh_dc.cpp's parity_fraction ALGORITHMICALLY -- same 8 cube
// diagonals first, same blocks-of-8 escalation, same 7:1 supermajority stop,
// same SplitMix64 + Shoemake per-vertex rotation from the same seed.
//
// Not bit-identical, and cannot be: float arithmetic contracts differently on
// the two devices, so a ray that grazes a triangle can land on either side and
// flip one vertex's parity. Measured on the bear, 14.09 M faces: the GPU and CPU
// outputs differ by 58 faces (0.0004%) and 0.02 open edges per 1000, with
// winding 100.00% either way. That is noise of the same order as the parity
// method's own sampling error, not a second algorithm -- but the CPU is the
// reference when the two are compared.
#include "tri_bvh.h"
#include <cstdio>
#include <cstdint>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>

namespace trellis {

namespace {

#define CU_OK(call) do { \
    const cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "  parity_gpu: %s failed: %s\n", #call, cudaGetErrorString(e_)); \
        return false; \
    } \
} while (0)

__constant__ float c_D8[24] = {
     0.5773503f,  0.5773503f,  0.5773503f,   0.5773503f,  0.5773503f, -0.5773503f,
     0.5773503f, -0.5773503f,  0.5773503f,   0.5773503f, -0.5773503f, -0.5773503f,
    -0.5773503f,  0.5773503f,  0.5773503f,  -0.5773503f,  0.5773503f, -0.5773503f,
    -0.5773503f, -0.5773503f,  0.5773503f,  -0.5773503f, -0.5773503f, -0.5773503f,
};

__device__ __forceinline__ uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ __forceinline__ float splitmix_unit(uint64_t& s) {
    return (float)((splitmix64(s) >> 11) * 1.11022302462515654042e-16);   // 2^-53
}

// Shoemake, identical to the host version -- same seed must give the same frame.
__device__ void random_rotation(uint64_t seed, float R[9]) {
    uint64_t s = seed * 0xD1B54A32D192ED03ull + 0x9E3779B97F4A7C15ull;
    const float u1 = splitmix_unit(s), u2 = splitmix_unit(s), u3 = splitmix_unit(s);
    const float s1 = sqrtf(1.0f - u1), s2 = sqrtf(u1);
    const float tau = 6.28318530717959f;
    // Precise sinf/cosf, NOT __sinf/__cosf: the fast intrinsics give a slightly
    // different rotation, and a slightly different direction can flip a grazing
    // ray's parity. Keeping the frames as close as float allows keeps the GPU
    // and CPU answers to within noise instead of merely close.
    const float x = s1 * sinf(tau * u2), y = s1 * cosf(tau * u2);
    const float z = s2 * sinf(tau * u3), w = s2 * cosf(tau * u3);
    R[0] = 1-2*(y*y+z*z); R[1] = 2*(x*y-z*w);   R[2] = 2*(x*z+y*w);
    R[3] = 2*(x*y+z*w);   R[4] = 1-2*(x*x+z*z); R[5] = 2*(y*z-x*w);
    R[6] = 2*(x*z-y*w);   R[7] = 2*(y*z+x*w);   R[8] = 1-2*(x*x+y*y);
}

// Crossing count along one ray. No nearest-hit pruning: parity needs EVERY
// crossing, so the traversal visits every node whose slab the ray meets.
__device__ int count_hits(const TriBvh::Node* nodes, const int32_t* prim,
                          const float* verts, const int32_t* faces,
                          const float org[3], const float dir[3], float max_t) {
    const float inv[3] = { 1.0f/dir[0], 1.0f/dir[1], 1.0f/dir[2] };
    int32_t stack[64];
    int sp = 0;
    int hits = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const int32_t ni = stack[--sp];
        const TriBvh::Node n = nodes[ni];
        // Slab test against this node.
        float t0 = 0.0f, t1 = max_t;
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            float a = (n.bmin[k] - org[k]) * inv[k];
            float b = (n.bmax[k] - org[k]) * inv[k];
            const float lo = fminf(a, b), hi = fmaxf(a, b);
            t0 = fmaxf(t0, lo); t1 = fminf(t1, hi);
        }
        if (t0 > t1) continue;
        if (n.count > 0) {
            for (int32_t i = 0; i < n.count; ++i) {
                const int32_t f = prim[n.left + i];
                const int32_t i0 = faces[3*f], i1 = faces[3*f+1], i2 = faces[3*f+2];
                const float ax = verts[3*i0], ay = verts[3*i0+1], az = verts[3*i0+2];
                const float e1x = verts[3*i1]-ax, e1y = verts[3*i1+1]-ay, e1z = verts[3*i1+2]-az;
                const float e2x = verts[3*i2]-ax, e2y = verts[3*i2+1]-ay, e2z = verts[3*i2+2]-az;
                // Moller-Trumbore, two-sided (|det|, not det > 0).
                const float pvx = dir[1]*e2z - dir[2]*e2y;
                const float pvy = dir[2]*e2x - dir[0]*e2z;
                const float pvz = dir[0]*e2y - dir[1]*e2x;
                const float det = e1x*pvx + e1y*pvy + e1z*pvz;
                if (fabsf(det) < 1e-20f) continue;
                const float invDet = 1.0f / det;
                const float tvx = org[0]-ax, tvy = org[1]-ay, tvz = org[2]-az;
                const float u = (tvx*pvx + tvy*pvy + tvz*pvz) * invDet;
                if (u < 0.0f || u > 1.0f) continue;
                const float qvx = tvy*e1z - tvz*e1y;
                const float qvy = tvz*e1x - tvx*e1z;
                const float qvz = tvx*e1y - tvy*e1x;
                const float v = (dir[0]*qvx + dir[1]*qvy + dir[2]*qvz) * invDet;
                if (v < 0.0f || u + v > 1.0f) continue;
                const float t = (e2x*qvx + e2y*qvy + e2z*qvz) * invDet;
                if (t > 1e-7f && t <= max_t) ++hits;
            }
            continue;
        }
        if (sp + 2 <= 64) { stack[sp++] = n.left + 1; stack[sp++] = n.left; }
    }
    return hits;
}

__global__ void parity_kernel(const TriBvh::Node* nodes, const int32_t* prim,
                              const float* verts, const int32_t* faces,
                              const float* pts, const uint64_t* seeds, int64_t n,
                              const float* dirs, int n_extra, float reach, float* out) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float p[3] = { pts[3*i], pts[3*i+1], pts[3*i+2] };
    float R[9];
    random_rotation(seeds[i], R);

    int odd = 0;
    #pragma unroll 1
    for (int k = 0; k < 8; ++k) {
        const float dx = c_D8[3*k], dy = c_D8[3*k+1], dz = c_D8[3*k+2];
        const float d[3] = { R[0]*dx + R[1]*dy + R[2]*dz,
                             R[3]*dx + R[4]*dy + R[5]*dz,
                             R[6]*dx + R[7]*dy + R[8]*dz };
        odd += count_hits(nodes, prim, verts, faces, p, d, reach) & 1;
    }
    if (odd == 0) { out[i] = 0.0f; return; }
    if (odd == 8) { out[i] = 1.0f; return; }

    int nn = 8;
    for (int b = 0; b < n_extra; b += 8) {
        const int e = min(n_extra, b + 8);
        for (int k = b; k < e; ++k) {
            const float dx = dirs[3*k], dy = dirs[3*k+1], dz = dirs[3*k+2];
            const float d[3] = { R[0]*dx + R[1]*dy + R[2]*dz,
                                 R[3]*dx + R[4]*dy + R[5]*dz,
                                 R[6]*dx + R[7]*dy + R[8]*dz };
            odd += count_hits(nodes, prim, verts, faces, p, d, reach) & 1;
        }
        nn = 8 + e;
        if (nn >= 16 && (odd * 8 <= nn || odd * 8 >= 7 * nn)) break;
    }
    out[i] = (float)odd / (float)nn;
}

}  // namespace

namespace {
// The uploaded BVH, kept between the remesh's two parity batches.
struct GpuBvh {
    const void* key = nullptr;
    TriBvh::Node* nodes = nullptr;
    int32_t* prim = nullptr;
    float* verts = nullptr;
    int32_t* faces = nullptr;
    size_t bytes = 0;
    void free_all() {
        cudaFree(nodes); cudaFree(prim); cudaFree(verts); cudaFree(faces);
        nodes = nullptr; prim = nullptr; verts = nullptr; faces = nullptr;
        key = nullptr; bytes = 0;
    }
};
GpuBvh g_bvh;
}  // namespace

// Hand the VRAM back once the remesh is done with the tree, so the stages that
// follow are not squeezed by a cache nobody is reading any more.
void parity_gpu_release() { if (g_bvh.key) g_bvh.free_all(); }

// Returns false (and leaves F untouched) when there is no usable device or the
// BVH will not fit, so the caller simply keeps the CPU path.
bool parity_fraction_gpu(const TriBvh& bvh, const std::vector<float>& pts,
                         const std::vector<uint64_t>& seeds,
                         const std::vector<float>& extra_dirs, float reach,
                         std::vector<float>& F) {
    const int64_t n = (int64_t)(pts.size() / 3);
    if (n == 0 || bvh.empty()) return false;

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return false;

    const size_t bytes_nodes = (size_t)bvh.n_nodes() * sizeof(TriBvh::Node);
    const size_t bytes_prim  = (size_t)bvh.n_prims() * sizeof(int32_t);
    const size_t bytes_verts = (size_t)bvh.n_verts() * 3 * sizeof(float);
    const size_t bytes_faces = (size_t)bvh.n_prims() * 3 * sizeof(int32_t);
    const size_t bytes_bvh   = bytes_nodes + bytes_prim + bytes_verts + bytes_faces;
    // Leave headroom: the caller may still hold decoder buffers, and a kernel
    // that OOMs mid-run is worse than never starting.
    if (bytes_bvh + ((size_t)256 << 20) > free_b) {
        printf("  parity_gpu: BVH needs %.2f GB, %.2f GB free -- staying on the CPU\n",
               bytes_bvh / 1073741824.0, free_b / 1073741824.0);
        return false;
    }

    float* d_dirs = nullptr; float* d_pts = nullptr; uint64_t* d_seeds = nullptr; float* d_out = nullptr;
    auto cleanup = [&] { cudaFree(d_dirs); cudaFree(d_pts); cudaFree(d_seeds); cudaFree(d_out); };

    // The remesh asks twice -- coarse lattice, then the vertices that could not
    // inherit -- so re-sending the tree would cost a second full upload of what
    // can be a gigabyte. Keyed on the node pointer, which changes whenever the
    // caller builds a different tree.
    const bool fresh = g_bvh.key != (const void*)bvh.nodes();
    if (fresh) {
        g_bvh.free_all();
        if (cudaMalloc(&g_bvh.nodes, bytes_nodes) != cudaSuccess ||
            cudaMalloc(&g_bvh.prim,  bytes_prim)  != cudaSuccess ||
            cudaMalloc(&g_bvh.verts, bytes_verts) != cudaSuccess ||
            cudaMalloc(&g_bvh.faces, bytes_faces) != cudaSuccess) { g_bvh.free_all(); return false; }
        if (cudaMemcpy(g_bvh.nodes, bvh.nodes(), bytes_nodes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(g_bvh.prim,  bvh.prims(), bytes_prim,  cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(g_bvh.verts, bvh.verts(), bytes_verts, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(g_bvh.faces, bvh.faces(), bytes_faces, cudaMemcpyHostToDevice) != cudaSuccess) {
            g_bvh.free_all(); return false;
        }
        g_bvh.key = (const void*)bvh.nodes();
        g_bvh.bytes = bytes_bvh;
    }
    TriBvh::Node* d_nodes = g_bvh.nodes; int32_t* d_prim = g_bvh.prim;
    float* d_verts = g_bvh.verts; int32_t* d_faces = g_bvh.faces;
    const int n_extra = (int)(extra_dirs.size() / 3);
    if (n_extra) {
        if (cudaMalloc(&d_dirs, extra_dirs.size()*sizeof(float)) != cudaSuccess ||
            cudaMemcpy(d_dirs, extra_dirs.data(), extra_dirs.size()*sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) { cleanup(); return false; }
    }

    // Chunk the points so peak memory stays bounded regardless of grid size.
    const int64_t chunk = 4 << 20;
    if (cudaMalloc(&d_pts,   (size_t)chunk*3*sizeof(float))   != cudaSuccess ||
        cudaMalloc(&d_seeds, (size_t)chunk*sizeof(uint64_t))  != cudaSuccess ||
        cudaMalloc(&d_out,   (size_t)chunk*sizeof(float))     != cudaSuccess) { cleanup(); return false; }

    F.resize((size_t)n);
    for (int64_t off = 0; off < n; off += chunk) {
        const int64_t m = (n - off) < chunk ? (n - off) : chunk;
        if (cudaMemcpy(d_pts, pts.data() + 3*off, (size_t)m*3*sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_seeds, seeds.data() + off, (size_t)m*sizeof(uint64_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) { cleanup(); return false; }
        const int block = 128;
        const int64_t grid = (m + block - 1) / block;
        parity_kernel<<<(unsigned)grid, block>>>(d_nodes, d_prim, d_verts, d_faces,
                                                 d_pts, d_seeds, m, d_dirs, n_extra, reach, d_out);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            fprintf(stderr, "  parity_gpu: kernel failed; falling back to the CPU\n");
            cleanup(); return false;
        }
        if (cudaMemcpy(F.data() + off, d_out, (size_t)m*sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) { cleanup(); return false; }
    }
    cleanup();
    printf("  parity_gpu: %lld vertices on the GPU (BVH %.2f GB, %s)\n",
           (long long)n, bytes_bvh / 1073741824.0, fresh ? "uploaded" : "cached");
    fflush(stdout);
    return true;
}

}  // namespace trellis

#endif  // __CUDACC__
