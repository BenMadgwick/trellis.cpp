// Shading frames: area-weighted vertex normals and UV-derived tangents.
//
// These MUST be shared between the texture bake and the GLB writer. A tangent-
// space normal map encodes a delta against a specific basis; if the baker and
// the runtime disagree about that basis by even a small rotation, lighting is
// subtly wrong everywhere and there is nothing in the output that looks like a
// bug. The writer used to compute normals privately, which was safe only
// because nothing else needed them.
#pragma once
#include <cstdint>
#include <vector>

namespace trellis {

// Area-weighted vertex normals, accumulated over POSITION-WELDED groups.
//
// The welding matters: xatlas duplicates vertices along UV seams, and per-copy
// normals see only their own chart's faces, so every chart border becomes a
// shading crease and each chart reads as its own facet. Welding by exact
// position reunites the copies, matching the reference (which computes normals
// before the UV split). Copies receive their group's normal, so `out` is
// indexed exactly like `verts`.
//
// Frame-agnostic: cross products rotate with the vertices, so calling this in
// TRELLIS space and rotating the result gives the same answer as calling it in
// the rotated frame (the glTF rotation (x,z,-y) is proper, det +1).
void vertex_normals(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                    std::vector<float>& out);

// Per-vertex tangents from UVs (Lengyel accumulation), Gram-Schmidt-orthogonalised
// against `nrm` and packed glTF-style as xyzw with w = +/-1 handedness, so the
// bitangent is w * cross(N, T). Accumulated over the same position-welded groups
// as vertex_normals for the same reason -- except that a UV seam is a genuine
// tangent discontinuity, so welding is by position AND uv-island: copies that
// disagree about UV keep their own tangent.
//
// Degenerate cases (zero-area UV triangle, tangent parallel to the normal) fall
// back to an arbitrary stable perpendicular; a normal map is meaningless there
// anyway, and the alternative is a NaN that poisons the whole vertex.
void vertex_tangents(const float* verts, int64_t V, const float* uv, const int32_t* faces, int64_t F,
                     const std::vector<float>& nrm, std::vector<float>& out4);

} // namespace trellis
