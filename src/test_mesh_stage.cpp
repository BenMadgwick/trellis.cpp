// Unit tests for the Interior mesh stage: TriBvh::count_hits, parity_fraction,
// the band-union field, and drop_duplicate_faces. CPU only, procedural inputs,
// no dumps needed.
//
//   trellis-test-mesh-stage [res]     (default 256; the synthetic-solid tests)
//
// The claims under test are the ones the design rests on, in the order they
// would break: rays count correctly, parity is bimodal, a solid loses its inner
// sheet, and a THIN SHEET DOES NOT -- that last one is the whole reason the
// field is a min() and not a product.
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "mesh_audit.h"
#include "uv_bake.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

using namespace trellis;

static int failures = 0;
static void check(bool ok, const char* what, const char* detail = "") {
    printf("  [%s] %s %s\n", ok ? " ok " : "FAIL", what, detail);
    if (!ok) ++failures;
}

// ---- procedural meshes ----------------------------------------------------

// Icosphere by subdividing an octahedron; outward-wound, closed.
static void make_sphere(int subdiv, float r, std::vector<float>& v, std::vector<int32_t>& f) {
    std::vector<float> p = {1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1, 0,0,-1};
    std::vector<int32_t> t = {0,2,4, 2,1,4, 1,3,4, 3,0,4, 2,0,5, 1,2,5, 3,1,5, 0,3,5};
    for (int s = 0; s < subdiv; ++s) {
        std::vector<int32_t> nt;
        std::unordered_map<uint64_t,int32_t> mid;
        auto edge = [&](int32_t a, int32_t b) -> int32_t {
            const uint64_t k = ((uint64_t)std::min(a,b) << 32) | (uint32_t)std::max(a,b);
            auto it = mid.find(k);
            if (it != mid.end()) return it->second;
            float m[3];
            for (int j = 0; j < 3; ++j) m[j] = (p[3*a+j] + p[3*b+j]) * 0.5f;
            const float l = std::sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
            const int32_t idx = (int32_t)(p.size() / 3);
            for (int j = 0; j < 3; ++j) p.push_back(m[j] / l);
            mid.emplace(k, idx);
            return idx;
        };
        for (size_t i = 0; i < t.size(); i += 3) {
            const int32_t a = t[i], b = t[i+1], c = t[i+2];
            const int32_t ab = edge(a,b), bc = edge(b,c), ca = edge(c,a);
            const int32_t tri[12] = {a,ab,ca, ab,b,bc, ca,bc,c, ab,bc,ca};
            nt.insert(nt.end(), tri, tri + 12);
        }
        t.swap(nt);
    }
    v.resize(p.size());
    for (size_t i = 0; i < p.size(); ++i) v[i] = p[i] * r;
    f = t;
}

// Axis-aligned box [c-h, c+h], outward-wound, 12 triangles.
static void make_box(const float c[3], const float h[3],
                     std::vector<float>& v, std::vector<int32_t>& f, bool flip = false) {
    const int base = (int)(v.size() / 3);
    for (int i = 0; i < 8; ++i)
        for (int k = 0; k < 3; ++k) v.push_back(c[k] + ((i >> k) & 1 ? h[k] : -h[k]));
    // Per axis, the two faces; winding chosen so normals point outward.
    static const int Q[6][4] = {{0,2,6,4},{1,5,7,3},{0,4,5,1},{2,3,7,6},{0,1,3,2},{4,6,7,5}};
    for (int q = 0; q < 6; ++q) {
        int a = Q[q][0], b = Q[q][1], c2 = Q[q][2], d = Q[q][3];
        if (flip) { const int s = b; b = d; d = s; }
        f.push_back(base+a); f.push_back(base+b); f.push_back(base+c2);
        f.push_back(base+a); f.push_back(base+c2); f.push_back(base+d);
    }
}

static size_t boundary_edges(const std::vector<int32_t>& faces) {
    std::unordered_map<uint64_t,int> e;
    const size_t F = faces.size() / 3;
    auto k = [](int a, int b){ if (a>b){int t=a;a=b;b=t;} return ((uint64_t)(uint32_t)a<<32)|(uint32_t)b; };
    for (size_t f = 0; f < F; ++f)
        for (int j = 0; j < 3; ++j) e[k(faces[3*f+j], faces[3*f+(j+1)%3])]++;
    size_t nb = 0;
    for (auto& kv : e) if (kv.second == 1) ++nb;
    return nb;
}

static double signed_volume(const std::vector<float>& v, const std::vector<int32_t>& f) {
    double vol = 0.0;
    for (size_t i = 0; i + 2 < f.size(); i += 3) {
        const float* A = &v[3*(size_t)f[i]];
        const float* B = &v[3*(size_t)f[i+1]];
        const float* C = &v[3*(size_t)f[i+2]];
        vol += ((double)A[0]*((double)B[1]*C[2] - (double)B[2]*C[1])
              - (double)A[1]*((double)B[0]*C[2] - (double)B[2]*C[0])
              + (double)A[2]*((double)B[0]*C[1] - (double)B[1]*C[0])) / 6.0;
    }
    return vol;
}

// ---- tests ----------------------------------------------------------------

// 1. count_hits parity on a closed sphere: odd from inside, even from outside,
//    and agreeing with a ray() restart loop on the same rays.
static void test_count_hits() {
    std::vector<float> v; std::vector<int32_t> f;
    make_sphere(2, 0.3f, v, f);
    const TriBvh bvh = TriBvh::build(v.data(), (int64_t)v.size()/3, f.data(), (int64_t)f.size()/3);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    int in_odd = 0, in_n = 0, out_even = 0, out_n = 0, agree = 0, cmp = 0;
    for (int i = 0; i < 2000 && (in_n < 1000 || out_n < 1000); ++i) {
        const float p[3] = {u(rng)*0.5f, u(rng)*0.5f, u(rng)*0.5f};
        const float r = std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
        if (r > 0.28f && r < 0.32f) continue;            // skip the surface itself
        const bool inside = r < 0.3f;
        if (inside ? in_n >= 1000 : out_n >= 1000) continue;
        float d[3] = {u(rng), u(rng), u(rng)};
        const float ld = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
        if (ld < 1e-3f) continue;
        for (int k = 0; k < 3; ++k) d[k] /= ld;
        const int n = bvh.count_hits(p, d, 10.0f);
        if (inside) { ++in_n; in_odd += (n & 1); } else { ++out_n; out_even += !(n & 1); }
        // Independent count via the restart loop the old parity_sign used.
        if (cmp < 200) {
            float org[3] = {p[0], p[1], p[2]};
            float left = 10.0f;
            int m = 0;
            for (int s = 0; s < 64; ++s) {
                const TriBvh::RayHit h = bvh.ray(org, d, left);
                if (h.face < 0) break;
                ++m;
                const float tt = h.t + 1e-6f;
                for (int k = 0; k < 3; ++k) org[k] += d[k] * tt;
                left -= tt;
                if (left <= 0.f) break;
            }
            agree += (m == n);
            ++cmp;
        }
    }
    char buf[160];
    std::snprintf(buf, sizeof(buf), "inside odd %d/%d, outside even %d/%d, matches ray() loop %d/%d",
                  in_odd, in_n, out_even, out_n, agree, cmp);
    check(in_odd == in_n && out_even == out_n && agree == cmp, "count_hits parity on a sphere", buf);
}

// 2. parity_fraction is bimodal, and intermediate ONLY in the plane of a hole.
//    parity_fraction is file-static in remesh_dc.cpp, so exercise the same
//    quantity through count_hits with the same 8+56 direction budget.
static void test_parity_fraction() {
    std::vector<float> v; std::vector<int32_t> f;
    make_sphere(3, 0.3f, v, f);
    // Cut a ~20 deg cap around +z: drop every face whose centroid is inside it.
    std::vector<int32_t> kept;
    for (size_t i = 0; i + 2 < f.size(); i += 3) {
        float c[3] = {0,0,0};
        for (int j = 0; j < 3; ++j) for (int k = 0; k < 3; ++k) c[k] += v[3*(size_t)f[i+j]+k] / 3.f;
        const float l = std::sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]);
        if (l > 1e-9f && c[2]/l > std::cos(20.0f * 3.14159265f / 180.0f)) continue;
        kept.push_back(f[i]); kept.push_back(f[i+1]); kept.push_back(f[i+2]);
    }
    f.swap(kept);
    const TriBvh bvh = TriBvh::build(v.data(), (int64_t)v.size()/3, f.data(), (int64_t)f.size()/3);
    auto frac = [&](const float p[3]) {
        std::mt19937 rng(999);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        int odd = 0, n = 0;
        while (n < 64) {
            float d[3] = {u(rng), u(rng), u(rng)};
            const float ld = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
            if (ld < 1e-3f) continue;
            for (int k = 0; k < 3; ++k) d[k] /= ld;
            odd += bvh.count_hits(p, d, 10.0f) & 1;
            ++n;
        }
        return (float)odd / (float)n;
    };
    const float centre[3] = {0, 0, 0};
    const float outside[3] = {0, 0, 0.6f};
    // In the plane of the hole: on the cap's own rim circle, z = 0.3*cos(20).
    const float rim[3] = {0, 0, 0.3f * std::cos(20.0f * 3.14159265f / 180.0f)};
    const float Fc = frac(centre), Fo = frac(outside), Fr = frac(rim);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "centre %.2f (>=0.9), 2r out %.2f (<=0.1), hole plane %.2f (0.3-0.7)",
                  Fc, Fo, Fr);
    check(Fc >= 0.9f && Fo <= 0.1f && Fr > 0.3f && Fr < 0.7f, "parity fraction on a holed sphere", buf);
}

// 3-5. The field itself, through remesh_narrow_band_dc.
static void test_remesh_modes(int res) {
    // 3. Solid box: Interior must halve the face count and stay closed.
    {
        std::vector<float> v; std::vector<int32_t> f;
        const float c[3] = {0,0,0}, h[3] = {0.03f, 0.03f, 0.03f};   // 60 mm cube
        make_box(c, h, v, f);
        const TriBvh bvh = TriBvh::build(v.data(), (int64_t)v.size()/3, f.data(), (int64_t)f.size()/3);
        const Mesh un = remesh_narrow_band_dc(v.data(), (int64_t)v.size()/3, f.data(), (int64_t)f.size()/3,
                                              bvh, res, 2, 0.0f, RemeshMode::Unsigned);
        const Mesh in = remesh_narrow_band_dc(v.data(), (int64_t)v.size()/3, f.data(), (int64_t)f.size()/3,
                                              bvh, res, 2, 0.0f, RemeshMode::Interior);
        // Not a face-count ratio: the two offset sheets of a box have areas in
        // the ratio (h+eps)^2 : (h-eps)^2, so "halves" only holds when eps is
        // small against the feature size (it is on the real assets: the dog's
        // unsigned count was exactly 2x its signed one). At res 256 eps is
        // 8 mm against a 30 mm half-extent, so the honest test is the
        // STRUCTURAL one -- is there any surface inside the box at all?
        //
        // The inner sheet sits at |x|inf = h - eps = 22 mm; the outer one is
        // never below h - eps/2 ~ 34.6 mm (its lowest points are on the corner
        // spheres). Anything under 26 mm is buried wall.
        auto min_linf = [](const Mesh& m) {
            double lo = 1e30;
            for (size_t i = 0; i + 2 < m.verts.size(); i += 3) {
                const double l = std::max(std::max(std::fabs(m.verts[i]), std::fabs(m.verts[i+1])),
                                          std::fabs(m.verts[i+2]));
                lo = std::min(lo, l);
            }
            return lo;
        };
        const double lo_un = min_linf(un), lo_in = min_linf(in);
        const size_t nb = boundary_edges(in.faces);
        const double vol = signed_volume(in.verts, in.faces);
        char buf[220];
        std::snprintf(buf, sizeof(buf),
                      "F %d -> %d; innermost surface %.1f mm -> %.1f mm (inner sheet at 22, outer >= 34.6); "
                      "boundary %zu, signed vol %+.3f cm3",
                      un.F(), in.F(), lo_un * 1000.0, lo_in * 1000.0, nb, vol * 1e6);
        check(lo_un < 0.026 && lo_in > 0.026 && in.F() < un.F() && nb == 0 && vol > 0.0,
              "solid box: Interior drops the inner sheet, closed, outward", buf);
    }
    // 4. Thin sheet: Interior must NOT drop it. A 1 mm slab has no parity
    //    interior, so the min() falls through to the unsigned term everywhere
    //    and the two outputs must agree. This is the case a signed field kills.
    {
        std::vector<float> v; std::vector<int32_t> f;
        const float c[3] = {0,0,0}, h[3] = {0.03f, 0.03f, 0.0005f};   // 60x60x1 mm
        make_box(c, h, v, f);
        const TriBvh bvh = TriBvh::build(v.data(), (int64_t)v.size()/3, f.data(), (int64_t)f.size()/3);
        const Mesh un = remesh_narrow_band_dc(v.data(), (int64_t)v.size()/3, f.data(), (int64_t)f.size()/3,
                                              bvh, res, 2, 0.0f, RemeshMode::Unsigned);
        const Mesh in = remesh_narrow_band_dc(v.data(), (int64_t)v.size()/3, f.data(), (int64_t)f.size()/3,
                                              bvh, res, 2, 0.0f, RemeshMode::Interior);
        const double ratio = un.F() ? (double)in.F() / (double)un.F() : 0.0;
        const size_t nb = boundary_edges(in.faces);
        char buf[200];
        std::snprintf(buf, sizeof(buf), "F %d vs %d (ratio %.3f, want ~1.0), boundary %zu", un.F(), in.F(), ratio, nb);
        check(ratio > 0.95 && ratio < 1.05 && nb == 0, "thin sheet: Interior keeps the bag", buf);
    }
    // 5. Hollow box: the inner wall is a separate closed component that no ray
    //    escapes, so the cull must leave exactly the outer one.
    {
        std::vector<float> v; std::vector<int32_t> f;
        const float c[3] = {0,0,0}, ho[3] = {0.03f,0.03f,0.03f}, hi[3] = {0.02f,0.02f,0.02f};
        make_box(c, ho, v, f);
        make_box(c, hi, v, f, /*flip=*/true);      // inner wall, normals inward
        const TriBvh bvh = TriBvh::build(v.data(), (int64_t)v.size()/3, f.data(), (int64_t)f.size()/3);
        Mesh in = remesh_narrow_band_dc(v.data(), (int64_t)v.size()/3, f.data(), (int64_t)f.size()/3,
                                        bvh, res, 2, 0.0f, RemeshMode::Interior);
        const int before = in.F();
        float bmin0[3] = {1e30f,1e30f,1e30f}, bmax0[3] = {-1e30f,-1e30f,-1e30f};
        for (size_t i = 0; i + 2 < in.verts.size(); i += 3)
            for (int k = 0; k < 3; ++k) {
                bmin0[k] = std::min(bmin0[k], in.verts[i+k]); bmax0[k] = std::max(bmax0[k], in.verts[i+k]);
            }
        cull_enclosed_components(in.verts, in.faces, 256, 100, false);
        float bmin[3] = {1e30f,1e30f,1e30f}, bmax[3] = {-1e30f,-1e30f,-1e30f};
        for (size_t i = 0; i + 2 < in.verts.size(); i += 3)
            for (int k = 0; k < 3; ++k) {
                bmin[k] = std::min(bmin[k], in.verts[i+k]); bmax[k] = std::max(bmax[k], in.verts[i+k]);
            }
        // The kept component must still span the OUTER box: the cull must not
        // have kept the inner wall and thrown the visible one away.
        const bool kept_outer = (bmax[0]-bmin[0]) > 0.9f * (bmax0[0]-bmin0[0]);
        char buf[200];
        std::snprintf(buf, sizeof(buf), "F %d -> %d, extent %.1f mm -> %.1f mm",
                      before, in.F(), (bmax0[0]-bmin0[0])*1000.f, (bmax[0]-bmin[0])*1000.f);
        check(in.F() > 0 && in.F() < before && kept_outer, "hollow box: cull drops the inner wall", buf);
    }
}

// 6. drop_duplicate_faces is orientation-agnostic: a face and its reverse
//    collapse to one, which is what parity needs (the pair is two crossings).
static void test_dedupe() {
    std::vector<int32_t> f = {0,1,2,  2,1,0,  0,1,2,  3,4,5};
    const int removed = drop_duplicate_faces(f);
    char buf[120];
    std::snprintf(buf, sizeof(buf), "removed %d, %zu faces left", removed, f.size()/3);
    check(removed == 2 && f.size() == 6, "drop_duplicate_faces: face + reverse -> one", buf);
}

int main(int argc, char** argv) {
    const int res = argc > 1 ? atoi(argv[1]) : 256;
    printf("mesh-stage tests (res %d)\n", res);
    test_count_hits();
    test_parity_fraction();
    test_dedupe();
    test_remesh_modes(res);
    printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
