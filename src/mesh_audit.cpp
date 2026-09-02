#include "mesh_audit.h"
#include "tri_bvh.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <functional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trellis {

namespace {

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

void fib_sphere(int n, std::vector<float>& out) {
    out.resize((size_t)n * 3);
    const float ga = 3.14159265358979f * (3.0f - 2.2360679774997896f);
    for (int i = 0; i < n; ++i) {
        const float z = 1.0f - (2.0f * i + 1.0f) / (float)n;
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float phi = (float)i * ga;
        out[3*i] = r * std::cos(phi); out[3*i+1] = r * std::sin(phi); out[3*i+2] = z;
    }
}

// Unnormalised face normal and twice the area.
inline void face_normal(const float* v, const int32_t* f, int64_t i, float n[3]) {
    const float* A = &v[3*(size_t)f[3*i]];
    const float* B = &v[3*(size_t)f[3*i+1]];
    const float* C = &v[3*(size_t)f[3*i+2]];
    const float e1[3] = {B[0]-A[0], B[1]-A[1], B[2]-A[2]};
    const float e2[3] = {C[0]-A[0], C[1]-A[1], C[2]-A[2]};
    n[0] = e1[1]*e2[2] - e1[2]*e2[1];
    n[1] = e1[2]*e2[0] - e1[0]*e2[2];
    n[2] = e1[0]*e2[1] - e1[1]*e2[0];
}

// Shared-vertex union-find over faces (the same relation drop_small_components
// uses, so the two agree on what a component is).
std::vector<int> face_components(const std::vector<float>& verts, const std::vector<int32_t>& faces,
                                 int& ncomp) {
    const int V = (int)(verts.size() / 3);
    const size_t F = faces.size() / 3;
    std::vector<int> par((size_t)V);
    for (int i = 0; i < V; ++i) par[i] = i;
    std::function<int(int)> find = [&](int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    auto uni = [&](int a, int b) { const int ra = find(a), rb = find(b); if (ra != rb) par[ra] = rb; };
    for (size_t f = 0; f < F; ++f) { uni(faces[3*f], faces[3*f+1]); uni(faces[3*f+1], faces[3*f+2]); }
    std::unordered_map<int,int> id;
    std::vector<int> comp(F, -1);
    for (size_t f = 0; f < F; ++f) {
        const int r = find(faces[3*f]);
        auto it = id.find(r);
        if (it == id.end()) it = id.emplace(r, (int)id.size()).first;
        comp[f] = it->second;
    }
    ncomp = (int)id.size();
    return comp;
}

}  // namespace

int64_t cull_enclosed_components(std::vector<float>& verts, std::vector<int32_t>& faces,
                                 int probes, int min_faces, bool verbose) {
    const size_t F = faces.size() / 3;
    if (F == 0) return 0;
    int ncomp = 0;
    const std::vector<int> comp = face_components(verts, faces, ncomp);

    std::vector<std::vector<int64_t>> cfaces((size_t)ncomp);
    std::vector<double> carea((size_t)ncomp, 0.0);
    for (size_t f = 0; f < F; ++f) {
        cfaces[(size_t)comp[f]].push_back((int64_t)f);
        float n[3];
        face_normal(verts.data(), faces.data(), (int64_t)f, n);
        carea[(size_t)comp[f]] += 0.5 * std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    }

    float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
    for (size_t i = 0; i + 2 < verts.size(); i += 3)
        for (int k = 0; k < 3; ++k) { bmin[k] = std::min(bmin[k], verts[i+k]); bmax[k] = std::max(bmax[k], verts[i+k]); }
    float diag = 0.f;
    for (int k = 0; k < 3; ++k) diag += (bmax[k]-bmin[k]) * (bmax[k]-bmin[k]);
    const float reach = 4.0f * std::sqrt(std::max(1e-12f, diag));

    const TriBvh bvh = TriBvh::build(verts.data(), (int64_t)verts.size()/3,
                                     faces.data(), (int64_t)faces.size()/3);
    std::vector<uint8_t> exposed((size_t)ncomp, 1);
    std::vector<int> escaped((size_t)ncomp, 0), probed((size_t)ncomp, 0);
    parallel_for(ncomp, [&](int64_t b, int64_t e) {
        for (int64_t c = b; c < e; ++c) {
            const std::vector<int64_t>& fl = cfaces[(size_t)c];
            if ((int64_t)fl.size() < min_faces) continue;   // left to drop_small_components
            const int64_t stride = std::max<int64_t>(1, (int64_t)fl.size() / std::max(1, probes));
            int esc = 0, np = 0;
            for (int64_t i = 0; i < (int64_t)fl.size(); i += stride) {
                const int64_t f = fl[(size_t)i];
                const float* A = &verts[3*(size_t)faces[3*f]];
                const float* B = &verts[3*(size_t)faces[3*f+1]];
                const float* C = &verts[3*(size_t)faces[3*f+2]];
                float n[3];
                face_normal(verts.data(), faces.data(), f, n);
                const float ln = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                if (ln <= 1e-30f) continue;
                for (int k = 0; k < 3; ++k) n[k] /= ln;
                // Start a hair off the surface along the ray, or the face the
                // probe came from is its own first hit.
                const float cen[3] = { (A[0]+B[0]+C[0])/3.f, (A[1]+B[1]+C[1])/3.f, (A[2]+B[2]+C[2])/3.f };
                for (int s = -1; s <= 1; s += 2) {
                    const float d[3] = { s*n[0], s*n[1], s*n[2] };
                    const float org[3] = { cen[0] + d[0]*1e-5f, cen[1] + d[1]*1e-5f, cen[2] + d[2]*1e-5f };
                    ++np;
                    if (bvh.count_hits(org, d, reach) == 0) ++esc;
                }
            }
            escaped[(size_t)c] = esc;
            probed[(size_t)c] = np;
            exposed[(size_t)c] = esc > 0;
        }
    });

    int64_t removed = 0;
    for (int c = 0; c < ncomp; ++c) {
        const bool big = (int64_t)cfaces[(size_t)c].size() >= min_faces;
        if (verbose && (big || !exposed[(size_t)c]))
            printf("  cull: comp %-4d faces=%-10zu area=%-10.2f cm2 exposed=%d escaped=%d/%d (%.0f%%)\n",
                   c, cfaces[(size_t)c].size(), carea[(size_t)c] * 1e4,
                   (int)exposed[(size_t)c], escaped[(size_t)c], probed[(size_t)c],
                   probed[(size_t)c] ? 100.0 * escaped[(size_t)c] / probed[(size_t)c] : 0.0);
        if (!exposed[(size_t)c]) removed += (int64_t)cfaces[(size_t)c].size();
    }

    // Escape rate on the LARGEST component: how much of it faces inward.
    //
    // On a single-walled closed surface one of each face's two probes points
    // outward and escapes unless the body occludes it, so the rate is high. A
    // component carrying inward-facing surface as well has both probes blocked
    // over that part, and the rate falls.
    //
    // Report it, but do NOT read it as "the decode is two-walled". It cannot
    // tell a decode artefact from a legitimately hollow object: the bear (a
    // two-walled bag, half its budget wasted) reads 27% and the jerry can (a
    // real hollow can, whose interior is genuine geometry) reads 24% -- yet
    // their 10 K visible AREA is 0.52 against 0.96. The decision signal is the
    // tier visible fraction (--audit-visible), which separates them cleanly;
    // this number only says inward-facing surface exists.
    int big_c = -1;
    for (int c = 0; c < ncomp; ++c)
        if (big_c < 0 || cfaces[(size_t)c].size() > cfaces[(size_t)big_c].size()) big_c = c;
    if (verbose && big_c >= 0 && probed[(size_t)big_c])
        printf("  cull: largest component escape rate %.0f%%%s\n",
               100.0 * escaped[(size_t)big_c] / probed[(size_t)big_c],
               2 * escaped[(size_t)big_c] < probed[(size_t)big_c]
                   ? "  (inward-facing surface present -- check --audit-visible at a tier"
                     " before concluding it is waste)" : "");

    if (removed == 0) {
        if (verbose) printf("  cull: every component is exposed, nothing removed\n");
        return 0;
    }
    // Never empty the mesh. If every component reads enclosed, the probe test
    // is wrong about something (a fully-enclosing outer shell, or a mesh so
    // small every probe is blocked) and deleting the lot would be catastrophic
    // where doing nothing is merely unhelpful.
    if (removed == (int64_t)F) {
        if (verbose) printf("  cull: ALL components read enclosed; keeping everything\n");
        return 0;
    }

    std::vector<int32_t> kf;
    kf.reserve(faces.size());
    for (size_t f = 0; f < F; ++f)
        if (exposed[(size_t)comp[f]])
            for (int k = 0; k < 3; ++k) kf.push_back(faces[3*f + k]);
    const int V = (int)(verts.size() / 3);
    std::vector<int> remap((size_t)V, -1);
    std::vector<float> nv;
    nv.reserve(verts.size());
    for (size_t i = 0; i < kf.size(); ++i) {
        const int v = kf[i];
        if (remap[(size_t)v] < 0) {
            remap[(size_t)v] = (int)(nv.size() / 3);
            nv.insert(nv.end(), &verts[3*(size_t)v], &verts[3*(size_t)v] + 3);
        }
        kf[i] = remap[(size_t)v];
    }
    verts.swap(nv);
    faces.swap(kf);
    if (verbose)
        printf("  cull: removed %lld enclosed faces -> F=%zu\n", (long long)removed, faces.size()/3);
    return removed;
}

VisibleAudit visible_fraction(const std::vector<float>& verts, const std::vector<int32_t>& faces,
                              int ndirs, int grid) {
    VisibleAudit a;
    const size_t F = faces.size() / 3;
    a.faces = (int64_t)F;
    if (F == 0) return a;

    std::vector<double> farea(F, 0.0);
    for (size_t f = 0; f < F; ++f) {
        float n[3];
        face_normal(verts.data(), faces.data(), (int64_t)f, n);
        farea[f] = 0.5 * std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        a.area += farea[f];
    }

    float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
    for (size_t i = 0; i + 2 < verts.size(); i += 3)
        for (int k = 0; k < 3; ++k) { bmin[k] = std::min(bmin[k], verts[i+k]); bmax[k] = std::max(bmax[k], verts[i+k]); }
    const float cen[3] = { (bmin[0]+bmax[0])*0.5f, (bmin[1]+bmax[1])*0.5f, (bmin[2]+bmax[2])*0.5f };
    float rad = 0.f;
    for (int k = 0; k < 3; ++k) rad += (bmax[k]-bmin[k]) * (bmax[k]-bmin[k]);
    rad = 0.5f * std::sqrt(std::max(1e-12f, rad)) * 1.05f;

    const TriBvh bvh = TriBvh::build(verts.data(), (int64_t)verts.size()/3,
                                     faces.data(), (int64_t)faces.size()/3);
    std::vector<float> dirs;
    fib_sphere(ndirs, dirs);
    // One byte per face, written by many threads. Distinct bytes, and the value
    // is only ever set to 1, so a benign race here costs nothing.
    std::vector<uint8_t> hit(F, 0);
    std::atomic<int64_t> landed{0};
    parallel_for(ndirs, [&](int64_t b, int64_t e) {
        int64_t local = 0;
        for (int64_t di = b; di < e; ++di) {
            const float d[3] = { dirs[3*di], dirs[3*di+1], dirs[3*di+2] };
            // Orthographic frame perpendicular to d.
            float u[3];
            if (std::fabs(d[0]) < 0.9f) { u[0] = 0; u[1] = -d[2]; u[2] = d[1]; }
            else                        { u[0] = -d[2]; u[1] = 0; u[2] = d[0]; }
            const float lu = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
            for (int k = 0; k < 3; ++k) u[k] /= lu;
            const float w[3] = { d[1]*u[2]-d[2]*u[1], d[2]*u[0]-d[0]*u[2], d[0]*u[1]-d[1]*u[0] };
            const float nd[3] = { -d[0], -d[1], -d[2] };   // fire back toward the object
            for (int py = 0; py < grid; ++py)
                for (int px = 0; px < grid; ++px) {
                    const float sx = (2.0f * (px + 0.5f) / grid - 1.0f) * rad;
                    const float sy = (2.0f * (py + 0.5f) / grid - 1.0f) * rad;
                    const float org[3] = { cen[0] + d[0]*rad*2.0f + u[0]*sx + w[0]*sy,
                                           cen[1] + d[1]*rad*2.0f + u[1]*sx + w[1]*sy,
                                           cen[2] + d[2]*rad*2.0f + u[2]*sx + w[2]*sy };
                    const TriBvh::RayHit h = bvh.ray(org, nd, rad * 8.0f);
                    if (h.face >= 0) { hit[(size_t)h.face] = 1; ++local; }
                }
        }
        if (local) landed.fetch_add(local, std::memory_order_relaxed);
    });

    a.rays_cast = (int64_t)ndirs * grid * grid;
    a.rays_hit = landed.load();
    for (size_t f = 0; f < F; ++f)
        if (hit[f]) { ++a.faces_hit; a.area_hit += farea[f]; }
    return a;
}

}  // namespace trellis
