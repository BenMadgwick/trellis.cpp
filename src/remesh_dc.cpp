#include "remesh_dc.h"
#include "tri_bvh.h"
#include <algorithm>
#include <atomic>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cmath>
#include <cstdio>
#include <functional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trellis {

#ifdef TRELLIS_HAVE_GPU_PARITY
// Provided by src/parity_gpu.cu. Returns false when there is no usable device or
// the BVH will not fit, in which case the CPU path below runs unchanged.
bool parity_fraction_gpu(const TriBvh& bvh, const std::vector<float>& pts,
                         const std::vector<uint64_t>& seeds,
                         const std::vector<float>& extra_dirs, float reach,
                         std::vector<float>& F);
void parity_gpu_release();
#endif

namespace {

inline int ctz64(uint64_t v) {
#ifdef _MSC_VER
    unsigned long i;
    _BitScanForward64(&i, v);
    return (int)i;
#else
    return __builtin_ctzll(v);
#endif
}

inline uint64_t key3(int x, int y, int z) {
    return ((uint64_t)(uint32_t)x << 42) | ((uint64_t)(uint32_t)y << 21) | (uint32_t)z;
}

void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    const int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> ts;
    const int64_t chunk = (n + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int64_t b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        ts.emplace_back(fn, b, e);
    }
    for (auto& t : ts) t.join();
}

inline uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
inline float splitmix_unit(uint64_t& s) { return (float)((splitmix64(s) >> 11) * 0x1.0p-53); }

// Spherical Fibonacci directions: near-uniform on the sphere for any N, and
// generated rather than tabulated so the ray budget is a runtime knob.
void fib_sphere(int n, std::vector<float>& out) {
    out.resize((size_t)n * 3);
    const float ga = 3.14159265358979f * (3.0f - 2.2360679774997896f);   // pi*(3-sqrt5)
    for (int i = 0; i < n; ++i) {
        const float z = 1.0f - (2.0f * i + 1.0f) / (float)n;
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float phi = (float)i * ga;
        out[3*i]   = r * std::cos(phi);
        out[3*i+1] = r * std::sin(phi);
        out[3*i+2] = z;
    }
}

// The 8 cube diagonals: the cheap first pass. Unanimity among 8 independent
// directions settles the vast majority of vertices, which are nowhere near a
// hole or a grazing configuration.
static const float D8[8][3] = {
    { 0.5773503f,  0.5773503f,  0.5773503f}, { 0.5773503f,  0.5773503f, -0.5773503f},
    { 0.5773503f, -0.5773503f,  0.5773503f}, { 0.5773503f, -0.5773503f, -0.5773503f},
    {-0.5773503f,  0.5773503f,  0.5773503f}, {-0.5773503f,  0.5773503f, -0.5773503f},
    {-0.5773503f, -0.5773503f,  0.5773503f}, {-0.5773503f, -0.5773503f, -0.5773503f},
};

// Uniformly random unit quaternion from a seed (Shoemake), as a rotation matrix.
// Every vertex gets its OWN rotation. Fixed directions shared across vertices
// make grazing errors CORRELATED: the five-direction sign miscounted along the
// whole of the dog's back, where the surface lies near-tangent to one of them,
// and the result was a hole down the spine rather than scattered specks. With a
// per-vertex rotation the same grazing hits one vertex and not its neighbour,
// so the error stays a speck -- and in Interior mode a speck is a closed bump.
void random_rotation(uint64_t seed, float R[9]) {
    uint64_t s = seed * 0xD1B54A32D192ED03ull + 0x9E3779B97F4A7C15ull;
    const float u1 = splitmix_unit(s), u2 = splitmix_unit(s), u3 = splitmix_unit(s);
    const float s1 = std::sqrt(1.0f - u1), s2 = std::sqrt(u1);
    const float tau = 6.28318530717959f;
    const float x = s1 * std::sin(tau * u2), y = s1 * std::cos(tau * u2);
    const float z = s2 * std::sin(tau * u3), w = s2 * std::cos(tau * u3);
    R[0] = 1-2*(y*y+z*z); R[1] = 2*(x*y-z*w);   R[2] = 2*(x*z+y*w);
    R[3] = 2*(x*y+z*w);   R[4] = 1-2*(x*x+z*z); R[5] = 2*(y*z-x*w);
    R[6] = 2*(x*z-y*w);   R[7] = 2*(y*z+x*w);   R[8] = 1-2*(x*x+y*y);
}

// Fraction of directions along which the input surface is crossed an odd number
// of times. 1 deep inside a closed region, 0 outside it, and intermediate only
// near a hole -- where the true answer is genuinely undefined and a cap belongs.
//
// Orientation-free by construction: it counts crossings and never reads a
// normal, which is the only reason it survives an input whose winding is a
// patchwork (see RemeshMode in the header).
float parity_fraction(const TriBvh& bvh, const float p[3], uint64_t seed,
                      const std::vector<float>& dirs, float reach, int64_t* rays_out) {
    float R[9];
    random_rotation(seed, R);
    auto rot = [&R](const float* d, float* o) {
        o[0] = R[0]*d[0] + R[1]*d[1] + R[2]*d[2];
        o[1] = R[3]*d[0] + R[4]*d[1] + R[5]*d[2];
        o[2] = R[6]*d[0] + R[7]*d[1] + R[8]*d[2];
    };
    int odd = 0;
    for (int i = 0; i < 8; ++i) {
        float d[3];
        rot(D8[i], d);
        odd += bvh.count_hits(p, d, reach) & 1;
    }
    if (odd == 0) { if (rays_out) *rays_out += 8; return 0.0f; }   // unanimously outside
    if (odd == 8) { if (rays_out) *rays_out += 8; return 1.0f; }   // unanimously inside
    // Escalate in blocks of 8 rather than jumping straight to the full budget.
    // The verdict is a comparison against F = 0.5, so once a supermajority is
    // established the remaining rays cannot change it -- and paying for them is
    // what made this stage expensive on a torn asset, where the 8 diagonals
    // disagree over large regions rather than only at holes.
    //
    // The stop rule (F <= 1/8 or F >= 7/8 with at least 16 samples) keeps the
    // full budget exactly where the design wants it: vertices whose true F is
    // near 0.5, i.e. in the plane of a hole, where a cap belongs either way.
    const int extra = (int)(dirs.size() / 3);
    int n = 8;
    for (int b = 0; b < extra; b += 8) {
        const int e = std::min(extra, b + 8);
        for (int i = b; i < e; ++i) {
            float d[3];
            rot(&dirs[3*i], d);
            odd += bvh.count_hits(p, d, reach) & 1;
        }
        n = 8 + e;
        if (n >= 16 && (odd * 8 <= n || odd * 8 >= 7 * n)) break;
    }
    if (rays_out) *rays_out += n;
    return (float)odd / (float)n;
}

}  // namespace

namespace {

// Angle-weighted pseudonormals (Baerentzen & Aanaes). The sign of a distance is
// only well defined against the right normal: at a vertex or an edge, the normal
// of whichever triangle the query happened to land on gives the wrong answer in
// concave regions -- and the closest point lands on a feature constantly here,
// because the triangles (~0.8 mm) are smaller than the grid cell (~1 mm).
struct PseudoNormals {
    std::vector<float> vn;   // [V*3], angle-weighted

    void build(const float* verts, int64_t V, const int32_t* faces, int64_t F) {
        vn.assign((size_t)V * 3, 0.f);
        for (int64_t f = 0; f < F; ++f) {
            const int32_t i0 = faces[3*f], i1 = faces[3*f+1], i2 = faces[3*f+2];
            const float* P[3] = { &verts[3*(size_t)i0], &verts[3*(size_t)i1], &verts[3*(size_t)i2] };
            const float e1[3] = {P[1][0]-P[0][0], P[1][1]-P[0][1], P[1][2]-P[0][2]};
            const float e2[3] = {P[2][0]-P[0][0], P[2][1]-P[0][1], P[2][2]-P[0][2]};
            float n[3] = { e1[1]*e2[2]-e1[2]*e2[1],
                           e1[2]*e2[0]-e1[0]*e2[2],
                           e1[0]*e2[1]-e1[1]*e2[0] };
            const float ln = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            if (ln <= 1e-30f) continue;
            for (int k = 0; k < 3; ++k) n[k] /= ln;
            const int32_t idx[3] = { i0, i1, i2 };
            for (int c = 0; c < 3; ++c) {
                const float* A = P[c];
                const float* B = P[(c+1)%3];
                const float* C = P[(c+2)%3];
                float u[3], v[3];
                for (int k = 0; k < 3; ++k) { u[k] = B[k]-A[k]; v[k] = C[k]-A[k]; }
                const float lu = std::sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
                const float lv = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
                if (lu <= 1e-30f || lv <= 1e-30f) continue;
                float d = (u[0]*v[0]+u[1]*v[1]+u[2]*v[2]) / (lu*lv);
                d = std::max(-1.f, std::min(1.f, d));
                const float w = std::acos(d);          // interior angle at this corner
                for (int k = 0; k < 3; ++k) vn[3*(size_t)idx[c]+k] += w * n[k];
            }
        }
        for (int64_t v = 0; v < V; ++v) {
            float* n = &vn[3*(size_t)v];
            const float l = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            if (l > 1e-30f) { for (int k = 0; k < 3; ++k) n[k] /= l; }
        }
    }

    // +1 outside, -1 inside by RAY PARITY: march to infinity and count crossings,
    // odd meaning inside. Exact for a closed region and, unlike a pseudonormal,
    // indifferent to how the surface is wound locally -- which matters because
    // this input carries 22,641 non-manifold edges against only 3,756 boundary
    // edges, so it is topologically messy far more than it is open.
    //
    // The direction is deliberately not axis-aligned: an axis ray through a
    // dense axis-aligned-ish mesh grazes edges constantly, and every graze is a
    // miscount.
    static float parity_sign(const TriBvh& bvh, const float p[3], float reach) {
        // Five directions, majority wins. A single ray is hostage to whatever it
        // happens to graze: on the dog it miscounted across the whole of the
        // back, where the surface lies near-tangent to one direction, and the
        // result was a hole along the spine. Grazing is direction-specific, so
        // independent directions disagree only where one of them is wrong.
        static const float D[5][3] = {
            {  0.5773503f,  0.5774085f,  0.5772921f },
            { -0.7071068f,  0.4082483f,  0.5773503f },
            {  0.3363364f, -0.8451543f,  0.4140393f },
            {  0.2672612f,  0.5345225f, -0.8017837f },
            { -0.4558423f, -0.5698029f, -0.6837635f },
        };
        int inside = 0;
        for (int d = 0; d < 5; ++d) {
            float org[3] = { p[0], p[1], p[2] };
            float left = reach;
            int crossings = 0;
            for (int i = 0; i < 64; ++i) {
                const TriBvh::RayHit h = bvh.ray(org, D[d], left);
                if (h.face < 0) break;
                ++crossings;
                // Step just past the hit. Too small and the same triangle is hit
                // again; too large and a thin wall is stepped over.
                const float t = h.t + 1e-6f;
                for (int k = 0; k < 3; ++k) org[k] += D[d][k] * t;
                left -= t;
                if (left <= 0.f) break;
            }
            if (crossings & 1) ++inside;
        }
        return inside >= 3 ? -1.f : 1.f;
    }

    // +1 outside, -1 inside, from the pseudonormal interpolated at the hit point.
    float sign_at(const float* verts, const int32_t* faces,
                  int32_t face, const float hp[3], const float p[3]) const {
        const int32_t i0 = faces[3*face], i1 = faces[3*face+1], i2 = faces[3*face+2];
        const float* A = &verts[3*(size_t)i0];
        const float* B = &verts[3*(size_t)i1];
        const float* C = &verts[3*(size_t)i2];
        float v0[3], v1[3], v2[3];
        for (int k = 0; k < 3; ++k) { v0[k] = B[k]-A[k]; v1[k] = C[k]-A[k]; v2[k] = hp[k]-A[k]; }
        const float d00 = v0[0]*v0[0]+v0[1]*v0[1]+v0[2]*v0[2];
        const float d01 = v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2];
        const float d11 = v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2];
        const float d20 = v2[0]*v0[0]+v2[1]*v0[1]+v2[2]*v0[2];
        const float d21 = v2[0]*v1[0]+v2[1]*v1[1]+v2[2]*v1[2];
        const float den = d00*d11 - d01*d01;
        float b1 = 0.f, b2 = 0.f;
        if (std::fabs(den) > 1e-30f) { b1 = (d11*d20 - d01*d21)/den; b2 = (d00*d21 - d01*d20)/den; }
        const float b0 = 1.f - b1 - b2;
        float n[3];
        for (int k = 0; k < 3; ++k)
            n[k] = b0*vn[3*(size_t)i0+k] + b1*vn[3*(size_t)i1+k] + b2*vn[3*(size_t)i2+k];
        float dot = 0.f;
        for (int k = 0; k < 3; ++k) dot += (p[k] - hp[k]) * n[k];
        return dot >= 0.f ? 1.f : -1.f;
    }
};

}  // namespace

Mesh remesh_narrow_band_dc(const float* iverts, int64_t iV, const int32_t* ifaces, int64_t iF,
                           const TriBvh& bvh, int res, int band, float project_back,
                           RemeshMode mode, int sign_rays, const char* sign_dump, int coarse) {
    Mesh out;
    if (iF == 0 || bvh.empty() || res <= 0) return out;
    const bool signed_field = mode == RemeshMode::Signed5;

    // Reference domain: the world cube is inflated by (res+3·band)/res so the
    // offset shell never touches the boundary; eps is the offset distance.
    const float scale = (float)(res + 3 * band) / (float)res;
    const float cell = scale / (float)res;
    const float eps = (float)band * cell;
    // Signed mode contours the zero level set; the band still sizes the active
    // region, it just no longer offsets the surface.
    const float lvl = signed_field ? 0.f : eps;
    PseudoNormals pn;
    if (signed_field) pn.build(iverts, iV, ifaces, iF);
    const float keep = 0.87f * cell;

    // Interior mode: kappa scales the parity term to [-eps, +eps], commensurate
    // with the UDF term at the band edge. Any value in [eps, 4*eps] gives the
    // same zero set -- kappa only shapes how a bump's crossing interpolates.
    const float kappa = 2.0f * eps;
    const float reach = 4.0f * scale;      // the inflated domain diagonal is < 2*scale
    std::vector<float> extra_dirs;
    if (mode == RemeshMode::Interior && sign_rays > 8) fib_sphere(sign_rays - 8, extra_dirs);
    std::atomic<int64_t> n_parity{0};
    std::atomic<int64_t> n_rays{0};
    int64_t n_inherited = 0, n_gpu_verts = 0;

    // Candidate cells: conservative dilation of every triangle's AABB by the
    // band-plus-crossing radius, marked in a res^3 bitset.
    const int64_t nbits = (int64_t)res * res * res;
    std::vector<uint64_t> cand((size_t)((nbits + 63) / 64), 0);
    auto bit_set = [&cand, res](int x, int y, int z) {
        const int64_t i = ((int64_t)x * res + y) * res + z;
        cand[(size_t)(i >> 6)] |= 1ull << (i & 63);
    };
    auto bit_get = [&cand, res](int x, int y, int z) -> bool {
        const int64_t i = ((int64_t)x * res + y) * res + z;
        return (cand[(size_t)(i >> 6)] >> (i & 63)) & 1;
    };
    // A cell is active iff UDF(center) < (band+0.87)·cell, and its closest
    // surface point lies inside some triangle-marked cell, so the per-axis
    // index distance is < band+1.37, i.e. ≤ band+1.
    const int dil = band + 1;
    {
        const int F = (int)iF;
        std::vector<std::vector<uint64_t>> parts;
        const int nt = std::max(1u, std::thread::hardware_concurrency());
        parts.assign(nt, {});
        std::vector<std::thread> ts;
        const int chunk = (F + nt - 1) / nt;
        for (int t = 0; t < nt; ++t) {
            const int b = t * chunk, e = std::min(F, b + chunk);
            if (b >= e) break;
            parts[t].assign(cand.size(), 0);
            ts.emplace_back([&, t, b, e]() {
                auto& bits = parts[t];
                auto setb = [&bits, res](int x, int y, int z) {
                    const int64_t i = ((int64_t)x * res + y) * res + z;
                    bits[(size_t)(i >> 6)] |= 1ull << (i & 63);
                };
                for (int f = b; f < e; ++f) {
                    float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
                    for (int j = 0; j < 3; ++j) {
                        const float* p = &iverts[3 * ifaces[3*f+j]];
                        for (int k = 0; k < 3; ++k) {
                            bmin[k] = std::min(bmin[k], p[k]);
                            bmax[k] = std::max(bmax[k], p[k]);
                        }
                    }
                    int c0[3], c1[3];
                    for (int k = 0; k < 3; ++k) {
                        c0[k] = std::max(0, (int)std::floor((bmin[k] / scale + 0.5f) * res) - dil);
                        c1[k] = std::min(res - 1, (int)std::floor((bmax[k] / scale + 0.5f) * res) + dil);
                    }
                    for (int x = c0[0]; x <= c1[0]; ++x)
                        for (int y = c0[1]; y <= c1[1]; ++y)
                            for (int z = c0[2]; z <= c1[2]; ++z) setb(x, y, z);
                }
            });
        }
        for (auto& th : ts) th.join();
        for (auto& bits : parts)
            if (!bits.empty())
                for (size_t i = 0; i < cand.size(); ++i) cand[i] |= bits[i];
    }

    // Active voxels: |UDF(center) - eps| < 0.87*cell (spec 27 §4.1).
    std::vector<int> acoord;
    {
        std::vector<int64_t> cand_cells;
        for (int64_t w = 0; w < (int64_t)cand.size(); ++w) {
            uint64_t bits = cand[w];
            while (bits) {
                const int b = ctz64(bits);
                bits &= bits - 1;
                cand_cells.push_back((w << 6) | b);
            }
        }
        std::vector<uint8_t> act(cand_cells.size(), 0);
        parallel_for((int64_t)cand_cells.size(), [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                const int64_t c = cand_cells[i];
                const int x = (int)(c / ((int64_t)res * res)), y = (int)((c / res) % res), z = (int)(c % res);
                const float p[3] = { ((x + 0.5f) / res - 0.5f) * scale,
                                     ((y + 0.5f) / res - 0.5f) * scale,
                                     ((z + 0.5f) / res - 0.5f) * scale };
                const TriBvh::Hit h = bvh.closest(p, lvl + keep);
                if (h.face < 0) continue;
                const float f = std::sqrt(h.dist2) - lvl;
                if (std::fabs(f) < keep) act[i] = 1;
            }
        });
        for (size_t i = 0; i < cand_cells.size(); ++i) {
            if (!act[i]) continue;
            const int64_t c = cand_cells[i];
            acoord.push_back((int)(c / ((int64_t)res * res)));
            acoord.push_back((int)((c / res) % res));
            acoord.push_back((int)(c % res));
        }
        cand.clear(); cand.shrink_to_fit();
    }
    const int64_t Na = (int64_t)acoord.size() / 3;
    if (Na < 100) {
        fprintf(stderr, "  remesh: only %lld active voxels; skipping remesh\n", (long long)Na);
        return out;
    }
    // Announce the scale BEFORE the field pass, which is the expensive part and
    // was previously silent from here until it finished. In Interior mode it
    // casts 8-64 rays per band vertex through a BVH over the whole input, and
    // that input can be enormous on a porous subject: measured, a 34.6 M-face
    // pallet of paving stones at res 768 ran over an hour without a word, while
    // a 14.7 M-face bear at res 1024 took 472 s. A stage that can take that long
    // must at least say what it is chewing on.
    printf("  remesh_dc: %lld active voxels over %lld input faces (%s, res %d, band %d)%s\n",
           (long long)Na, (long long)iF,
           mode == RemeshMode::Interior ? "INTERIOR" : (mode == RemeshMode::Signed5 ? "SIGNED5" : "UNSIGNED"),
           res, band,
           mode == RemeshMode::Interior && iF > 20000000
               ? "  <- large porous input; the parity pass will be slow, consider --remesh-mode unsigned"
               : "");
    fflush(stdout);

    std::unordered_map<uint64_t, int> vox;
    vox.reserve((size_t)Na * 2);
    for (int64_t i = 0; i < Na; ++i) vox.emplace(key3(acoord[3*i], acoord[3*i+1], acoord[3*i+2]), (int)i);

    // f = UDF - eps at the grid VERTICES (corner mapping v/res, spec 27 §4.3).
    std::vector<int> vcoord;
    std::unordered_map<uint64_t, int> vmap;
    vmap.reserve((size_t)Na * 3);
    for (int64_t i = 0; i < Na; ++i)
        for (int dx = 0; dx < 2; ++dx) for (int dy = 0; dy < 2; ++dy) for (int dz = 0; dz < 2; ++dz) {
            const int x = acoord[3*i] + dx, y = acoord[3*i+1] + dy, z = acoord[3*i+2] + dz;
            if (vmap.emplace(key3(x, y, z), (int)(vcoord.size() / 3)).second) {
                vcoord.push_back(x); vcoord.push_back(y); vcoord.push_back(z);
            }
        }
    const int64_t Nv = (int64_t)vcoord.size() / 3;
    std::vector<float> fvert((size_t)Nv);
    // R1 diagnostic: x, y, z, udf, F per vertex where parity was evaluated.
    // F = -1 marks "not evaluated" (the UDF term decided it) and is filtered out
    // when the file is written.
    std::vector<float> dump;
    if (sign_dump && mode == RemeshMode::Interior) dump.assign((size_t)Nv * 5, -1.0f);

    // ---- pass 1: UDF at every grid vertex ---------------------------------
    // Split out from the parity work so the coarse-to-fine pass below can look
    // at a neighbour's answer before deciding whether to pay for its own.
    std::vector<float> udf((size_t)Nv);        // signed against lvl, as before
    std::vector<uint8_t> want((size_t)Nv, 0);  // needs a parity answer at all
    std::vector<uint8_t> isfar((size_t)Nv, 0);
    parallel_for(Nv, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const float p[3] = { ((float)vcoord[3*i]   / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+1] / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+2] / res - 0.5f) * scale };
            // Crossing edges always have their far endpoint under eps+cell
            // (f changes at most one cell-length per edge), so this bound
            // never clips a value that feeds the crossing interpolation.
            const TriBvh::Hit h = bvh.closest(p, lvl + 2 * cell);
            const bool far = h.face < 0;
            const float d = far ? 2 * cell : std::sqrt(h.dist2) - lvl;
            udf[i] = d;
            isfar[i] = far;
            if (mode != RemeshMode::Interior) {
                // Past the search radius. Active cells hug the surface and their
                // corners sit within a cell of it, so this is the far field:
                // positive either way.
                fvert[i] = far ? 2 * cell : (signed_field ? d * PseudoNormals::parity_sign(bvh, p, reach) : d);
                continue;
            }
            // Deep inside the band the UDF term already binds (f < 0 whichever
            // way parity goes), so skip the rays entirely. This is most of the
            // vertices, and it is what keeps the stage at unsigned-mode cost.
            if (!far && d < -cell) { fvert[i] = d; continue; }
            want[i] = 1;
        }
    });

    if (mode == RemeshMode::Interior) {
        // ---- passes 2 and 3: coarse-to-fine parity ------------------------
        // A GPU does this pass 20-100x faster when one is available: every
        // vertex is independent, every ray is independent, and BVH traversal is
        // what the hardware exists for. The CPU path below stays the reference
        // -- identical directions, escalation and per-vertex seeds -- so a
        // fallback is a slowdown and never a change in output.
        // F is 0 or 1 across almost the whole volume and only varies near a
        // hole, so evaluating every vertex from scratch pays full price for an
        // answer its neighbours already knew. Evaluate a lattice of every
        // `coarse`-th vertex first, then let a fine vertex inherit when all
        // eight surrounding coarse samples agree.
        //
        // Conservative on purpose. A fine vertex inherits ONLY when all eight
        // corners exist AND are unanimous; a missing corner (the band's edge,
        // which is exactly where thin features live) forces a full evaluation.
        // And a feature thin enough to hide between coarse samples is thinner
        // than the band, where the UDF term decides the field anyway and parity
        // is not consulted at all.
        std::vector<float> Fv((size_t)Nv, -1.0f);
        std::atomic<int64_t> n_coarse{0}, n_inherit{0};
        const int cs = std::max(1, coarse);

        auto eval = [&](int64_t i, int64_t* rays) {
            const float p[3] = { ((float)vcoord[3*i]   / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+1] / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+2] / res - 0.5f) * scale };
            const uint64_t sd = key3(vcoord[3*i], vcoord[3*i+1], vcoord[3*i+2]);
            return parity_fraction(bvh, p, sd, extra_dirs, reach, rays);
        };
        auto on_lattice = [&](int64_t i) {
            return (vcoord[3*i] % cs) == 0 && (vcoord[3*i+1] % cs) == 0 && (vcoord[3*i+2] % cs) == 0;
        };

        // One batch runner for both passes: the GPU wants a large contiguous
        // batch, and the CPU fallback walks the identical list, so neither path
        // can drift from the other.
        std::atomic<int64_t> n_gpu{0};
        auto run_batch = [&](const std::vector<int64_t>& todo) {
            if (todo.empty()) return;
#ifdef TRELLIS_HAVE_GPU_PARITY
            {
                std::vector<float> pts((size_t)todo.size() * 3);
                std::vector<uint64_t> sds((size_t)todo.size());
                for (size_t t = 0; t < todo.size(); ++t) {
                    const int64_t i = todo[t];
                    pts[3*t]   = ((float)vcoord[3*i]   / res - 0.5f) * scale;
                    pts[3*t+1] = ((float)vcoord[3*i+1] / res - 0.5f) * scale;
                    pts[3*t+2] = ((float)vcoord[3*i+2] / res - 0.5f) * scale;
                    sds[t] = key3(vcoord[3*i], vcoord[3*i+1], vcoord[3*i+2]);
                }
                std::vector<float> gF;
                if (parity_fraction_gpu(bvh, pts, sds, extra_dirs, reach, gF) && gF.size() == todo.size()) {
                    for (size_t t = 0; t < todo.size(); ++t) Fv[todo[t]] = gF[t];
                    n_coarse.fetch_add((int64_t)todo.size(), std::memory_order_relaxed);
                    n_gpu.fetch_add((int64_t)todo.size(), std::memory_order_relaxed);
                    return;
                }
            }
#endif
            parallel_for((int64_t)todo.size(), [&](int64_t b, int64_t e) {
                int64_t local_rays = 0, local_n = 0;
                for (int64_t t = b; t < e; ++t) {
                    Fv[todo[t]] = eval(todo[t], &local_rays);
                    ++local_n;
                }
                if (local_n) { n_coarse.fetch_add(local_n, std::memory_order_relaxed);
                               n_rays.fetch_add(local_rays, std::memory_order_relaxed); }
            });
        };

        std::vector<int64_t> todo;
        for (int64_t i = 0; i < Nv; ++i)
            if (want[i] && (cs == 1 || on_lattice(i))) todo.push_back(i);
        run_batch(todo);

        if (cs > 1) {
            // Decide inherit-or-evaluate for every fine vertex FIRST, then send
            // the survivors through the same batch runner. Deciding is cheap
            // (eight map lookups); evaluating is not, and batching it is what
            // lets the GPU see one large launch instead of a trickle.
            std::vector<uint8_t> need((size_t)Nv, 0);
            parallel_for(Nv, [&](int64_t b, int64_t e) {
                int64_t local_inh = 0;
                for (int64_t i = b; i < e; ++i) {
                    if (!want[i] || Fv[i] >= 0.0f) continue;
                    // The eight lattice corners of the coarse cell holding this vertex.
                    const int base[3] = { (vcoord[3*i]   / cs) * cs,
                                          (vcoord[3*i+1] / cs) * cs,
                                          (vcoord[3*i+2] / cs) * cs };
                    int inside = 0, seen = 0;
                    for (int dx = 0; dx <= 1; ++dx)
                        for (int dy = 0; dy <= 1; ++dy)
                            for (int dz = 0; dz <= 1; ++dz) {
                                auto it = vmap.find(key3(base[0] + dx*cs, base[1] + dy*cs, base[2] + dz*cs));
                                if (it == vmap.end()) continue;
                                const float f = Fv[it->second];
                                if (f < 0.0f) continue;      // corner had no parity answer
                                ++seen;
                                inside += f > 0.5f;
                            }
                    // Four is the practical floor, not eight. The vertices that
                    // need parity form a SHELL two or three cells thick, not a
                    // volume, so a stride-2 cell almost never has all eight
                    // corners inside it -- measured, an 8-corner rule inherited
                    // 0.4% and cost more in extra passes than it saved. The
                    // corners that do exist are the tangential ones, along the
                    // band, which is the direction F is coherent in anyway.
                    if (seen >= 4 && (inside == 0 || inside == seen)) {
                        Fv[i] = inside ? 1.0f : 0.0f;        // unanimous: inherit, cast nothing
                        ++local_inh;
                        continue;
                    }
                    need[i] = 1;
                }
                if (local_inh) n_inherit.fetch_add(local_inh, std::memory_order_relaxed);
            });
            std::vector<int64_t> todo2;
            for (int64_t i = 0; i < Nv; ++i) if (need[i]) todo2.push_back(i);
            run_batch(todo2);
        }
        n_gpu_verts = n_gpu.load();
#ifdef TRELLIS_HAVE_GPU_PARITY
        // The dual-contouring and cleanup below need no BVH, and the caller has
        // texture stages after that; give the VRAM back now.
        if (n_gpu_verts) parity_gpu_release();
#endif

        n_parity.store(n_coarse.load() + n_inherit.load());
        n_inherited = n_inherit.load();

        // ---- pass 4: assemble the field -----------------------------------
        parallel_for(Nv, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                if (!want[i]) continue;
                const float F = Fv[i] < 0.0f ? 0.0f : Fv[i];
                if (!dump.empty()) {
                    dump[5*i] = ((float)vcoord[3*i]   / res - 0.5f) * scale;
                    dump[5*i+1] = ((float)vcoord[3*i+1] / res - 0.5f) * scale;
                    dump[5*i+2] = ((float)vcoord[3*i+2] / res - 0.5f) * scale;
                    dump[5*i+3] = isfar[i] ? lvl + 2 * cell : udf[i] + lvl;
                    dump[5*i+4] = F;
                }
                // The far field is only "positive either way" for an UNSIGNED
                // field. It is exactly where a solid's interior lives, and the
                // reason the eps level set doubles back on itself; parity is
                // what tells the two apart.
                if (isfar[i]) { fvert[i] = F > 0.5f ? -eps : 2 * cell; continue; }
                fvert[i] = std::min(udf[i], kappa * (0.5f - F));
            }
        });
    }
    if (!dump.empty()) {
        FILE* df = fopen(sign_dump, "wb");
        if (!df) fprintf(stderr, "  remesh_dc: cannot write --sign-dump %s\n", sign_dump);
        else {
            int64_t n = 0;
            for (int64_t i = 0; i < Nv; ++i)
                if (dump[5*i+4] >= 0.0f) { fwrite(&dump[5*i], 4, 5, df); ++n; }
            fclose(df);
            printf("  remesh_dc: --sign-dump wrote %lld x (x,y,z,udf,F) -> %s\n", (long long)n, sign_dump);
        }
    }
    auto fval = [&](int x, int y, int z) -> float {
        auto it = vmap.find(key3(x, y, z));
        return it == vmap.end() ? 1e9f : fvert[it->second];
    };

    // Dual vertices: plain mean of edge crossings, cell-center fallback; per
    // voxel, ownership of the 3 "far" edges records crossing direction
    // (spec 27 §4.4).
    std::vector<float> dual((size_t)Na * 3);
    std::vector<int8_t> owned((size_t)Na * 3, 0);
    parallel_for(Na, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
            double sum[3] = {0, 0, 0};
            int cnt = 0;
            for (int axis = 0; axis < 3; ++axis)
                for (int u = 0; u < 2; ++u) for (int v = 0; v < 2; ++v) {
                    int a0[3] = {vx, vy, vz}, a1[3];
                    a0[(axis + 1) % 3] += u;
                    a0[(axis + 2) % 3] += v;
                    a1[0] = a0[0]; a1[1] = a0[1]; a1[2] = a0[2];
                    a1[axis] += 1;
                    const float v1 = fval(a0[0], a0[1], a0[2]);
                    const float v2 = fval(a1[0], a1[1], a1[2]);
                    const bool c12 = v1 < 0 && v2 >= 0, c21 = v1 >= 0 && v2 < 0;
                    if (c12 || c21) {
                        const float t = -v1 / (v2 - v1);
                        double pt[3] = {(double)a0[0], (double)a0[1], (double)a0[2]};
                        pt[axis] += t;
                        for (int k = 0; k < 3; ++k) sum[k] += pt[k];
                        ++cnt;
                    }
                    if (u == 1 && v == 1) owned[3*i + axis] = c12 ? 1 : (c21 ? -1 : 0);
                }
            if (cnt) for (int k = 0; k < 3; ++k) dual[3*i+k] = (float)(sum[k] / cnt);
            else { dual[3*i] = vx + 0.5f; dual[3*i+1] = vy + 0.5f; dual[3*i+2] = vz + 0.5f; }
        }
    });

    // Quad assembly per owned crossing edge (spec 27 §4.5); winding from the
    // crossing direction. The reference's "planar diagonal" selection is a
    // latent no-op upstream (always diagonal q0-q2 unless triangle q0q1q2 is
    // exactly degenerate) — reproduced faithfully here as split 1 always.
    static const int OFF[3][4][3] = {
        {{0,0,0}, {0,0,1}, {0,1,1}, {0,1,0}},
        {{0,0,0}, {1,0,0}, {1,0,1}, {0,0,1}},
        {{0,0,0}, {0,1,0}, {1,1,0}, {1,0,0}},
    };
    std::vector<int32_t> qfaces;
    qfaces.reserve((size_t)Na * 6);
    std::vector<uint8_t> used((size_t)Na, 0);
    // Every crossing whose four surrounding cells are not all active loses its
    // quad and leaves a boundary edge. In Unsigned mode this is rare, because
    // the active predicate tracks the UDF term's zero set exactly. In Interior
    // mode the parity term can also cross, outside the band, over a hole wider
    // than 2*eps -- and those cells were never marked active. Counting the drops
    // says directly whether that is where the open edges come from (R3).
    int64_t dropped_quads = 0, total_quads = 0;
    for (int64_t i = 0; i < Na; ++i) {
        const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
        for (int axis = 0; axis < 3; ++axis) {
            const int dir = owned[3*i + axis];
            if (!dir) continue;
            int q[4];
            bool ok = true;
            for (int k = 0; k < 4 && ok; ++k) {
                auto it = vox.find(key3(vx + OFF[axis][k][0], vy + OFF[axis][k][1], vz + OFF[axis][k][2]));
                if (it == vox.end()) ok = false;
                else q[k] = it->second;
            }
            ++total_quads;
            if (!ok) { ++dropped_quads; continue; }
            static const int S1N[6] = {0, 1, 2, 0, 2, 3};
            static const int S1P[6] = {0, 2, 1, 0, 3, 2};
            const int* sp = dir > 0 ? S1P : S1N;
            for (int k = 0; k < 6; ++k) qfaces.push_back(q[sp[k]]);
            for (int k = 0; k < 4; ++k) used[q[k]] = 1;
        }
    }
    if (qfaces.empty()) return out;

    // Compact used dual vertices; map back to world coordinates.
    std::vector<int32_t> remap((size_t)Na, -1);
    int nv2 = 0;
    for (int64_t i = 0; i < Na; ++i)
        if (used[i]) {
            remap[i] = nv2++;
            out.verts.push_back((dual[3*i]   / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+1] / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+2] / res - 0.5f) * scale);
        }
    out.faces.resize(qfaces.size());
    for (size_t k = 0; k < qfaces.size(); ++k) out.faces[k] = remap[qfaces[k]];

    // Project the dual vertices back onto the input surface (reference:
    // remesh_project=0.9, o_voxel/postprocess.py::to_glb -> remeshing.py §8).
    // Dual contouring places each vertex at the plain MEAN of its cell's edge
    // crossings, on the eps-offset shell — so sharp features come out rounded and
    // the shell keeps its own offset noise. Lerping each vertex back toward its
    // closest point on the original surface restores those features. The BVH's
    // closest point is exactly the reference's barycentric interpolation of the
    // hit triangle, so no uvw round-trip is needed.
    if (project_back > 0.0f) {
        const int64_t NV = (int64_t)out.verts.size() / 3;
        std::atomic<int64_t> missed{0};
        parallel_for(NV, [&](int64_t b, int64_t e) {
            int64_t local = 0;
            for (int64_t i = b; i < e; ++i) {
                float* v = &out.verts[3 * i];
                const float p[3] = {v[0], v[1], v[2]};
                // Vertices sit on the eps shell, so the surface is ~eps away; the
                // same bound the field pass uses is comfortably sufficient.
                const TriBvh::Hit h = bvh.closest(p, lvl + 2 * cell);
                if (h.face < 0) { ++local; continue; }   // no hit in range: leave as-is
                for (int k = 0; k < 3; ++k) v[k] = p[k] - project_back * (p[k] - h.point[k]);
            }
            if (local) missed.fetch_add(local, std::memory_order_relaxed);
        });
        if (missed.load())
            printf("  remesh_dc: project_back %.2f (%lld/%lld vertices had no hit in range)\n",
                   project_back, (long long)missed.load(), (long long)NV);
    }

    const char* mname = mode == RemeshMode::Interior ? "INTERIOR"
                      : (mode == RemeshMode::Signed5 ? "SIGNED5" : "UNSIGNED");
    printf("  remesh_dc: %lld active voxels -> V=%d F=%d (%s %s=%.4g, project_back=%.2f)\n",
           (long long)Na, out.V(), out.F(), mname, signed_field ? "lvl" : "eps",
           lvl, project_back);
    if (dropped_quads)
        printf("  remesh_dc: %lld/%lld crossing quads dropped (%.3f%%) -- corners outside the "
               "active set; each leaves up to 4 boundary edges\n",
               (long long)dropped_quads, (long long)total_quads,
               total_quads ? 100.0 * (double)dropped_quads / (double)total_quads : 0.0);
    if (mode == RemeshMode::Interior) {
        const int64_t np = n_parity.load(), ni = n_inherited, ev = np - ni;
        printf("  remesh_dc: parity at %lld/%lld grid vertices (%.1f%%); %lld evaluated, "
               "%lld inherited from the coarse lattice (%.1f%% free, stride %d)\n",
               (long long)np, (long long)Nv, Nv ? 100.0 * (double)np / (double)Nv : 0.0,
               (long long)ev, (long long)ni, np ? 100.0 * (double)ni / (double)np : 0.0,
               std::max(1, coarse));
        const int64_t cpu_ev = ev - n_gpu_verts;
        printf("  remesh_dc: %lld evaluated on the GPU, %lld on the CPU (%lld rays, %.1f per CPU "
               "vertex, max %d), kappa=%.4g\n",
               (long long)n_gpu_verts, (long long)cpu_ev, (long long)n_rays.load(),
               cpu_ev ? (double)n_rays.load() / (double)cpu_ev : 0.0,
               std::max(8, sign_rays), kappa);
    }
    fflush(stdout);
    return out;
}

}  // namespace trellis
