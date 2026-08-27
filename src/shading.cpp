#include "shading.h"

#include <array>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace trellis {
namespace {

struct Vec3Hash {
    size_t operator()(const std::array<float,3>& k) const {
        size_t h = 1469598103934665603ull;
        const unsigned char* p = (const unsigned char*)k.data();
        for (int i = 0; i < 12; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        return h;
    }
};

struct Vec5Hash {
    size_t operator()(const std::array<float,5>& k) const {
        size_t h = 1469598103934665603ull;
        const unsigned char* p = (const unsigned char*)k.data();
        for (int i = 0; i < 20; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        return h;
    }
};

// Map every vertex to the lowest index sharing its exact position.
void weld_by_position(const float* verts, int64_t V, std::vector<int64_t>& rep) {
    std::unordered_map<std::array<float,3>, int64_t, Vec3Hash> first;
    first.reserve((size_t)V);
    rep.resize((size_t)V);
    for (int64_t i = 0; i < V; ++i) {
        const std::array<float,3> k{verts[3*i], verts[3*i+1], verts[3*i+2]};
        rep[i] = first.emplace(k, i).first->second;
    }
}

inline void normalize3(float* v, float fx, float fy, float fz) {
    const float l = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (l > 1e-20f) { v[0] /= l; v[1] /= l; v[2] /= l; }
    else { v[0] = fx; v[1] = fy; v[2] = fz; }
}

} // namespace

void vertex_normals(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                    std::vector<float>& out) {
    out.assign((size_t)V*3, 0.f);
    if (V <= 0 || F <= 0) return;

    std::vector<int64_t> rep;
    weld_by_position(verts, V, rep);

    // Unnormalised cross product = 2*area*unit_normal, so summing raw cross
    // products IS the area weighting.
    for (int64_t f = 0; f < F; ++f) {
        const int64_t a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        float e1[3], e2[3], n[3];
        for (int k = 0; k < 3; ++k) { e1[k] = verts[3*b+k]-verts[3*a+k]; e2[k] = verts[3*c+k]-verts[3*a+k]; }
        n[0] = e1[1]*e2[2]-e1[2]*e2[1];
        n[1] = e1[2]*e2[0]-e1[0]*e2[2];
        n[2] = e1[0]*e2[1]-e1[1]*e2[0];
        for (int j = 0; j < 3; ++j) {
            const int64_t v = rep[faces[3*f+j]];
            for (int k = 0; k < 3; ++k) out[3*v+k] += n[k];
        }
    }
    for (int64_t i = 0; i < V; ++i)
        if (rep[i] == i) normalize3(&out[3*i], 0.f, 0.f, 1.f);
    for (int64_t i = 0; i < V; ++i) {
        const int64_t r = rep[i];
        if (r != i) std::memcpy(&out[3*i], &out[3*r], 3*sizeof(float));
    }
}

void vertex_tangents(const float* verts, int64_t V, const float* uv, const int32_t* faces, int64_t F,
                     const std::vector<float>& nrm, std::vector<float>& out4) {
    out4.assign((size_t)V*4, 0.f);
    if (V <= 0 || F <= 0 || (int64_t)nrm.size() < V*3) return;

    // Weld by position AND uv: a UV seam is a real tangent discontinuity, so the
    // two sides of a seam must keep separate tangents even though they share a
    // normal. (vertex_normals deliberately does the opposite.)
    std::vector<int64_t> rep((size_t)V);
    {
        std::unordered_map<std::array<float,5>, int64_t, Vec5Hash> first;
        first.reserve((size_t)V);
        for (int64_t i = 0; i < V; ++i) {
            const std::array<float,5> k{verts[3*i], verts[3*i+1], verts[3*i+2], uv[2*i], uv[2*i+1]};
            rep[i] = first.emplace(k, i).first->second;
        }
    }

    std::vector<float> tan((size_t)V*3, 0.f), bit((size_t)V*3, 0.f);
    for (int64_t f = 0; f < F; ++f) {
        const int64_t a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        float e1[3], e2[3];
        for (int k = 0; k < 3; ++k) { e1[k] = verts[3*b+k]-verts[3*a+k]; e2[k] = verts[3*c+k]-verts[3*a+k]; }
        const float du1 = uv[2*b]-uv[2*a],   dv1 = uv[2*b+1]-uv[2*a+1];
        const float du2 = uv[2*c]-uv[2*a],   dv2 = uv[2*c+1]-uv[2*a+1];
        const float det = du1*dv2 - du2*dv1;
        if (std::fabs(det) < 1e-20f) continue;   // degenerate in UV: contributes nothing
        const float r = 1.f / det;
        float T[3], B[3];
        for (int k = 0; k < 3; ++k) {
            T[k] = (e1[k]*dv2 - e2[k]*dv1) * r;
            B[k] = (e2[k]*du1 - e1[k]*du2) * r;
        }
        for (int j = 0; j < 3; ++j) {
            const int64_t v = rep[faces[3*f+j]];
            for (int k = 0; k < 3; ++k) { tan[3*v+k] += T[k]; bit[3*v+k] += B[k]; }
        }
    }

    for (int64_t i = 0; i < V; ++i) {
        const int64_t r = rep[i];
        const float* N = &nrm[3*i];
        const float* Ta = &tan[3*r];
        const float* Ba = &bit[3*r];

        // Gram-Schmidt: T' = normalize(T - N*dot(N,T))
        const float d = N[0]*Ta[0] + N[1]*Ta[1] + N[2]*Ta[2];
        float t[3] = { Ta[0]-N[0]*d, Ta[1]-N[1]*d, Ta[2]-N[2]*d };
        float len = std::sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2]);
        if (len <= 1e-12f) {
            // No usable UV gradient (unmapped vertex, or T parallel to N).
            // Any stable perpendicular will do -- the map carries no signal here.
            const float ax = std::fabs(N[0]) < 0.9f ? 1.f : 0.f;
            t[0] = ax - N[0]*(N[0]*ax); t[1] = -N[1]*(N[0]*ax); t[2] = -N[2]*(N[0]*ax);
            len = std::sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2]);
            if (len <= 1e-12f) { t[0]=0.f; t[1]=1.f; t[2]=0.f; len=1.f; }
        }
        for (int k = 0; k < 3; ++k) t[k] /= len;

        // Handedness: does the accumulated bitangent agree with cross(N,T)?
        const float cx = N[1]*t[2]-N[2]*t[1], cy = N[2]*t[0]-N[0]*t[2], cz = N[0]*t[1]-N[1]*t[0];
        const float w = (cx*Ba[0] + cy*Ba[1] + cz*Ba[2]) < 0.f ? -1.f : 1.f;

        out4[4*i+0]=t[0]; out4[4*i+1]=t[1]; out4[4*i+2]=t[2]; out4[4*i+3]=w;
    }
}

} // namespace trellis
