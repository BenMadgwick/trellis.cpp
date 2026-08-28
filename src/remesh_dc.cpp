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
                           bool signed_field) {
    Mesh out;
    if (iF == 0 || bvh.empty() || res <= 0) return out;

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
    parallel_for(Nv, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const float p[3] = { ((float)vcoord[3*i]   / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+1] / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+2] / res - 0.5f) * scale };
            // Crossing edges always have their far endpoint under eps+cell
            // (f changes at most one cell-length per edge), so this bound
            // never clips a value that feeds the crossing interpolation.
            const TriBvh::Hit h = bvh.closest(p, lvl + 2 * cell);
            if (h.face < 0) {
                // Past the search radius. Active cells hug the surface and their
                // corners sit within a cell of it, so this is the far field:
                // positive either way.
                fvert[i] = 2 * cell;
            } else {
                const float d = std::sqrt(h.dist2) - lvl;
                fvert[i] = signed_field
                    ? d * PseudoNormals::parity_sign(bvh, p, 4.0f * scale)
                    : d;
            }
        }
    });
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
            if (!ok) continue;
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

    printf("  remesh_dc: %lld active voxels -> V=%d F=%d (%s=%.4g, project_back=%.2f)\n",
           (long long)Na, out.V(), out.F(), signed_field ? "SIGNED lvl" : "eps",
           lvl, project_back);
    fflush(stdout);
    return out;
}

}  // namespace trellis
