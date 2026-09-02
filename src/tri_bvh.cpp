#include "tri_bvh.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace trellis {

namespace {

inline float dot3(const float* a, const float* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

// Ericson, Real-Time Collision Detection 5.1.5: closest point on triangle to p.
void closest_on_tri(const float* p, const float* a, const float* b, const float* c, float* out) {
    float ab[3], ac[3], ap[3];
    for (int k = 0; k < 3; ++k) { ab[k] = b[k]-a[k]; ac[k] = c[k]-a[k]; ap[k] = p[k]-a[k]; }
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.f && d2 <= 0.f) { std::memcpy(out, a, 12); return; }
    float bp[3];
    for (int k = 0; k < 3; ++k) bp[k] = p[k]-b[k];
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.f && d4 <= d3) { std::memcpy(out, b, 12); return; }
    const float vc = d1*d4 - d3*d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        const float v = d1 / (d1 - d3);
        for (int k = 0; k < 3; ++k) out[k] = a[k] + v*ab[k];
        return;
    }
    float cp[3];
    for (int k = 0; k < 3; ++k) cp[k] = p[k]-c[k];
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.f && d5 <= d6) { std::memcpy(out, c, 12); return; }
    const float vb = d5*d2 - d1*d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        const float w = d2 / (d2 - d6);
        for (int k = 0; k < 3; ++k) out[k] = a[k] + w*ac[k];
        return;
    }
    const float va = d3*d6 - d5*d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        for (int k = 0; k < 3; ++k) out[k] = b[k] + w*(c[k]-b[k]);
        return;
    }
    const float denom = 1.f / (va + vb + vc);
    const float v = vb * denom, w = vc * denom;
    for (int k = 0; k < 3; ++k) out[k] = a[k] + ab[k]*v + ac[k]*w;
}

inline float box_dist2(const float* p, const float* bmin, const float* bmax) {
    float d2 = 0.f;
    for (int k = 0; k < 3; ++k) {
        const float v = p[k] < bmin[k] ? bmin[k] - p[k] : (p[k] > bmax[k] ? p[k] - bmax[k] : 0.f);
        d2 += v * v;
    }
    return d2;
}

}  // namespace

TriBvh TriBvh::build(const float* verts, int64_t V, const int32_t* faces, int64_t F) {
    (void)V;
    TriBvh t;
    t.verts_ = verts;
    t.faces_ = faces;
    if (F == 0) return t;
    t.prim_.resize((size_t)F);
    std::vector<float> cent((size_t)F * 3);
    for (int64_t f = 0; f < F; ++f) {
        t.prim_[f] = (int32_t)f;
        for (int k = 0; k < 3; ++k)
            cent[3*f+k] = (verts[3*faces[3*f]+k] + verts[3*faces[3*f+1]+k] + verts[3*faces[3*f+2]+k]) / 3.f;
    }
    t.nodes_.reserve((size_t)F * 2);

    struct Span { int32_t node, begin, end; };
    std::vector<Span> stack;
    t.nodes_.push_back({});
    stack.push_back({0, 0, (int32_t)F});
    while (!stack.empty()) {
        const Span s = stack.back(); stack.pop_back();
        Node& n0 = t.nodes_[s.node];
        float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
        for (int32_t i = s.begin; i < s.end; ++i) {
            const int32_t f = t.prim_[i];
            for (int j = 0; j < 3; ++j) {
                const float* v = &verts[3*faces[3*f+j]];
                for (int k = 0; k < 3; ++k) {
                    bmin[k] = std::min(bmin[k], v[k]);
                    bmax[k] = std::max(bmax[k], v[k]);
                }
            }
        }
        std::memcpy(n0.bmin, bmin, 12);
        std::memcpy(n0.bmax, bmax, 12);
        const int32_t cnt = s.end - s.begin;
        if (cnt <= 4) {
            n0.left = s.begin;
            n0.count = cnt;
            continue;
        }
        int axis = 0;
        float ext[3] = {bmax[0]-bmin[0], bmax[1]-bmin[1], bmax[2]-bmin[2]};
        if (ext[1] > ext[axis]) axis = 1;
        if (ext[2] > ext[axis]) axis = 2;
        const int32_t mid = s.begin + cnt / 2;
        std::nth_element(t.prim_.begin() + s.begin, t.prim_.begin() + mid, t.prim_.begin() + s.end,
                         [&cent, axis](int32_t a, int32_t b) { return cent[3*a+axis] < cent[3*b+axis]; });
        const int32_t lc = (int32_t)t.nodes_.size();
        t.nodes_[s.node].left = lc;
        t.nodes_[s.node].count = 0;
        t.nodes_.push_back({});
        t.nodes_.push_back({});
        stack.push_back({lc, s.begin, mid});
        stack.push_back({lc + 1, mid, s.end});
    }
    return t;
}

TriBvh::Hit TriBvh::closest(const float p[3], float max_dist) const {
    Hit hit;
    if (nodes_.empty()) return hit;
    hit.dist2 = max_dist * max_dist;
    struct Entry { float d2; int32_t node; };
    Entry stack[64];
    int sp = 0;
    stack[sp++] = {box_dist2(p, nodes_[0].bmin, nodes_[0].bmax), 0};
    while (sp > 0) {
        const Entry e = stack[--sp];
        if (e.d2 >= hit.dist2) continue;
        const Node& n = nodes_[e.node];
        if (n.count > 0) {
            for (int32_t i = 0; i < n.count; ++i) {
                const int32_t f = prim_[n.left + i];
                float q[3];
                closest_on_tri(p, &verts_[3*faces_[3*f]], &verts_[3*faces_[3*f+1]], &verts_[3*faces_[3*f+2]], q);
                const float dx = q[0]-p[0], dy = q[1]-p[1], dz = q[2]-p[2];
                const float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < hit.dist2) {
                    hit.dist2 = d2;
                    hit.face = f;
                    std::memcpy(hit.point, q, 12);
                }
            }
            continue;
        }
        const int32_t l = n.left, r = n.left + 1;
        float dl = box_dist2(p, nodes_[l].bmin, nodes_[l].bmax);
        float dr = box_dist2(p, nodes_[r].bmin, nodes_[r].bmax);
        int32_t first = l, second = r;
        if (dr < dl) { std::swap(dl, dr); first = r; second = l; }
        if (sp + 2 <= 64) {
            if (dr < hit.dist2) stack[sp++] = {dr, second};
            if (dl < hit.dist2) stack[sp++] = {dl, first};
        }
    }
    return hit;
}

TriBvh::RayHit TriBvh::ray(const float org[3], const float dir[3], float max_t) const {
    RayHit hit;
    if (nodes_.empty()) return hit;
    hit.t = max_t;

    // Slab test, precomputed reciprocals. Infinities are intentional: a zero
    // component gives +/-inf here and the comparisons below still order
    // correctly, which is the standard branch-free slab formulation.
    const float inv[3] = {1.0f / dir[0], 1.0f / dir[1], 1.0f / dir[2]};
    auto box_t = [&](const float bmin[3], const float bmax[3]) -> float {
        float t0 = 0.0f, t1 = hit.t;
        for (int k = 0; k < 3; ++k) {
            float a = (bmin[k] - org[k]) * inv[k];
            float b = (bmax[k] - org[k]) * inv[k];
            if (a > b) { const float s = a; a = b; b = s; }
            if (a > t0) t0 = a;
            if (b < t1) t1 = b;
            if (t0 > t1) return 1e30f;
        }
        return t0;
    };

    struct Entry { float t; int32_t node; };
    Entry stack[64];
    int sp = 0;
    const float t0root = box_t(nodes_[0].bmin, nodes_[0].bmax);
    if (t0root >= hit.t) return hit;
    stack[sp++] = {t0root, 0};
    while (sp > 0) {
        const Entry e = stack[--sp];
        if (e.t >= hit.t) continue;              // a closer surface was already found
        const Node& n = nodes_[e.node];
        if (n.count > 0) {
            for (int32_t i = 0; i < n.count; ++i) {
                const int32_t f = prim_[n.left + i];
                // Moller-Trumbore. Two-sided: an inverted sheet still counts as
                // a surface you can see, which is the whole point here.
                const float* a = &verts_[3*faces_[3*f]];
                const float* b = &verts_[3*faces_[3*f+1]];
                const float* c = &verts_[3*faces_[3*f+2]];
                float e1[3], e2[3], pv[3];
                for (int k = 0; k < 3; ++k) { e1[k] = b[k]-a[k]; e2[k] = c[k]-a[k]; }
                pv[0] = dir[1]*e2[2] - dir[2]*e2[1];
                pv[1] = dir[2]*e2[0] - dir[0]*e2[2];
                pv[2] = dir[0]*e2[1] - dir[1]*e2[0];
                const float det = dot3(e1, pv);
                if (std::fabs(det) < 1e-20f) continue;   // ray parallel to the triangle
                const float invDet = 1.0f / det;
                float tv[3], qv[3];
                for (int k = 0; k < 3; ++k) tv[k] = org[k]-a[k];
                const float u = dot3(tv, pv) * invDet;
                if (u < 0.0f || u > 1.0f) continue;
                qv[0] = tv[1]*e1[2] - tv[2]*e1[1];
                qv[1] = tv[2]*e1[0] - tv[0]*e1[2];
                qv[2] = tv[0]*e1[1] - tv[1]*e1[0];
                const float v = dot3(dir, qv) * invDet;
                if (v < 0.0f || u + v > 1.0f) continue;
                const float t = dot3(e2, qv) * invDet;
                if (t > 1e-7f && t < hit.t) { hit.t = t; hit.face = f; }
            }
            continue;
        }
        const int32_t l = n.left, r = n.left + 1;
        float tl = box_t(nodes_[l].bmin, nodes_[l].bmax);
        float tr = box_t(nodes_[r].bmin, nodes_[r].bmax);
        int32_t first = l, second = r;
        if (tr < tl) { const float s = tl; tl = tr; tr = s; first = r; second = l; }
        if (sp + 2 <= 64) {
            if (tr < hit.t) stack[sp++] = {tr, second};   // farther child first
            if (tl < hit.t) stack[sp++] = {tl, first};
        }
    }
    return hit;
}

int TriBvh::count_hits(const float org[3], const float dir[3], float max_t) const {
    if (nodes_.empty()) return 0;
    int hits = 0;

    // Same branch-free slab test as ray(), but the far bound is FIXED at max_t
    // instead of shrinking to the nearest hit found so far -- we want every
    // crossing, not the first.
    const float inv[3] = {1.0f / dir[0], 1.0f / dir[1], 1.0f / dir[2]};
    auto box_t = [&](const float bmin[3], const float bmax[3]) -> float {
        float t0 = 0.0f, t1 = max_t;
        for (int k = 0; k < 3; ++k) {
            float a = (bmin[k] - org[k]) * inv[k];
            float b = (bmax[k] - org[k]) * inv[k];
            if (a > b) { const float s = a; a = b; b = s; }
            if (a > t0) t0 = a;
            if (b < t1) t1 = b;
            if (t0 > t1) return 1e30f;
        }
        return t0;
    };

    // Deeper than ray()'s 64: that one prunes hard as hit.t collapses, this one
    // descends the whole overlapping subtree. A median split is balanced, so
    // depth is ~log2(F/4) -- 23 at 32 M faces -- but a dropped node here would
    // silently flip a parity rather than merely miss a hit, so the headroom is
    // worth 512 bytes of stack.
    struct Entry { float t; int32_t node; };
    Entry stack[128];
    int sp = 0;
    if (box_t(nodes_[0].bmin, nodes_[0].bmax) >= 1e30f) return 0;
    stack[sp++] = {0.0f, 0};
    while (sp > 0) {
        const Entry e = stack[--sp];
        const Node& n = nodes_[e.node];
        if (n.count > 0) {
            for (int32_t i = 0; i < n.count; ++i) {
                const int32_t f = prim_[n.left + i];
                // Moller-Trumbore, two-sided (|det| test rather than det > 0).
                const float* a = &verts_[3*faces_[3*f]];
                const float* b = &verts_[3*faces_[3*f+1]];
                const float* c = &verts_[3*faces_[3*f+2]];
                float e1[3], e2[3], pv[3];
                for (int k = 0; k < 3; ++k) { e1[k] = b[k]-a[k]; e2[k] = c[k]-a[k]; }
                pv[0] = dir[1]*e2[2] - dir[2]*e2[1];
                pv[1] = dir[2]*e2[0] - dir[0]*e2[2];
                pv[2] = dir[0]*e2[1] - dir[1]*e2[0];
                const float det = dot3(e1, pv);
                if (std::fabs(det) < 1e-20f) continue;   // ray parallel to the triangle
                const float invDet = 1.0f / det;
                float tv[3], qv[3];
                for (int k = 0; k < 3; ++k) tv[k] = org[k]-a[k];
                const float u = dot3(tv, pv) * invDet;
                if (u < 0.0f || u > 1.0f) continue;
                qv[0] = tv[1]*e1[2] - tv[2]*e1[1];
                qv[1] = tv[2]*e1[0] - tv[0]*e1[2];
                qv[2] = tv[0]*e1[1] - tv[1]*e1[0];
                const float v = dot3(dir, qv) * invDet;
                if (v < 0.0f || u + v > 1.0f) continue;
                const float t = dot3(e2, qv) * invDet;
                if (t > 1e-7f && t <= max_t) ++hits;
            }
            continue;
        }
        const int32_t l = n.left, r = n.left + 1;
        const float tl = box_t(nodes_[l].bmin, nodes_[l].bmax);
        const float tr = box_t(nodes_[r].bmin, nodes_[r].bmax);
        // No ordering: without nearest-hit pruning, visit order cannot change
        // the count, so push whichever children overlap and move on.
        if (sp + 2 <= 128) {
            if (tr < 1e30f) stack[sp++] = {tr, r};
            if (tl < 1e30f) stack[sp++] = {tl, l};
        }
    }
    return hits;
}

}  // namespace trellis
