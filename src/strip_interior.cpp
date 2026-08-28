#include "strip_interior.h"
#include "tri_bvh.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace trellis {

// Which sheet is the outside? Answer it the way the question is actually posed:
// the outside is the region of space you can reach from infinity, and the outer
// sheet is the part of the surface that bounds it.
//
// Two approaches were tried and discarded first, both for structural reasons
// rather than tuning, and both are worth recording so they are not retried:
//
//   Ray sampling.  Fire parallel rays from many directions and keep whatever is
//   hit first. Coverage improves only as ~sqrt(rays), because a face seen at a
//   grazing angle presents almost no area. Measured on a 27.6M-face shell,
//   quadrupling the rays cut the open boundary by 2.7x; a clean result would
//   have needed order 1e10 rays.
//
//   Seed-and-grow across folds.  Seed from rays, then grow over the surface and
//   refuse to cross a sharp dihedral, on the theory that the rim where the shell
//   wraps back on itself is a ~180 degree fold. It is - but only in aggregate.
//   At 27.6M faces the triangles are sub-millimetre and the rim is spread over
//   dozens of them, each turning a few degrees, so nothing reads as a fold
//   locally and growth walks around every rim into the interior. It swallowed
//   91.6% of the mesh.
//
// The flood fill has neither failure mode: no sampling, and no local decisions.
// The mesh is watertight coming out of remesh_dc (boundary_edges=0), so voxels
// the surface passes through seal the exterior off from everything inside.

namespace {

// Exact triangle/cell overlap (separating-axis test), NOT the triangle's
// bounding box.
//
// The box version marked up to 8 cells where a triangle touches 1 or 2 -- the
// remesh's edges are ~0.83 mm against a 0.98 mm cell, so every triangle is
// sub-cell and its box is 2x2x2. That fattened each wall until the two walls of
// the 3.9 mm offset shell merged into one blob, which is why the depth
// histogram never showed two populations, why no trough was ever found on a
// two-sheet asset, and why no probe distance separates the walls. Over-marking
// is safe for sealing and fatal for telling inside from outside.
//
// Still conservative -- a cell the triangle touches at all is marked -- so the
// shell stays watertight and the flood cannot leak.
bool tri_cell_overlap(const float bc[3], const float h[3],
                      const float a[3], const float b[3], const float c[3]) {
    float v0[3], v1[3], v2[3];
    for (int k = 0; k < 3; ++k) { v0[k] = a[k]-bc[k]; v1[k] = b[k]-bc[k]; v2[k] = c[k]-bc[k]; }
    float e0[3], e1[3], e2[3];
    for (int k = 0; k < 3; ++k) { e0[k] = v1[k]-v0[k]; e1[k] = v2[k]-v1[k]; e2[k] = v0[k]-v2[k]; }

    // Separated along `ax`? Project the triangle and the box and look for a gap.
    auto sep = [&](const float ax[3]) -> bool {
        const float p0 = ax[0]*v0[0] + ax[1]*v0[1] + ax[2]*v0[2];
        const float p1 = ax[0]*v1[0] + ax[1]*v1[1] + ax[2]*v1[2];
        const float p2 = ax[0]*v2[0] + ax[1]*v2[1] + ax[2]*v2[2];
        const float mn = std::min(p0, std::min(p1, p2));
        const float mx = std::max(p0, std::max(p1, p2));
        const float rad = std::fabs(ax[0])*h[0] + std::fabs(ax[1])*h[1] + std::fabs(ax[2])*h[2];
        return mn > rad || mx < -rad;
    };

    // the nine edge x box-axis cross products
    const float* E[3] = { e0, e1, e2 };
    for (int i = 0; i < 3; ++i) {
        const float* e = E[i];
        const float ax0[3] = { 0.f,   -e[2],  e[1] };
        const float ax1[3] = { e[2],   0.f,  -e[0] };
        const float ax2[3] = { -e[1],  e[0],  0.f  };
        if (sep(ax0) || sep(ax1) || sep(ax2)) return false;
    }
    // the three box face normals
    const float X[3] = {1,0,0}, Y[3] = {0,1,0}, Z[3] = {0,0,1};
    if (sep(X) || sep(Y) || sep(Z)) return false;
    // the triangle's own plane
    const float n[3] = { e0[1]*e1[2]-e0[2]*e1[1],
                         e0[2]*e1[0]-e0[0]*e1[2],
                         e0[0]*e1[1]-e0[1]*e1[0] };
    if (sep(n)) return false;
    return true;
}

void voxelize(const std::vector<float>& verts, const std::vector<int32_t>& faces,
              const float org[3], float cell, int R, std::vector<uint8_t>& g) {
    const size_t F = faces.size() / 3;
    const float inv = 1.0f / cell;
    const float h[3] = { 0.5f*cell, 0.5f*cell, 0.5f*cell };
    for (size_t f = 0; f < F; ++f) {
        const float* A = &verts[3*(size_t)faces[3*f+0]];
        const float* B = &verts[3*(size_t)faces[3*f+1]];
        const float* C = &verts[3*(size_t)faces[3*f+2]];
        int lo[3], hi[3];
        for (int k = 0; k < 3; ++k) {
            const float mn = std::min(A[k], std::min(B[k], C[k]));
            const float mx = std::max(A[k], std::max(B[k], C[k]));
            lo[k] = std::max(0, std::min(R-1, (int)std::floor((mn - org[k]) * inv)));
            hi[k] = std::max(0, std::min(R-1, (int)std::floor((mx - org[k]) * inv)));
        }
        for (int z = lo[2]; z <= hi[2]; ++z)
            for (int y = lo[1]; y <= hi[1]; ++y)
                for (int x = lo[0]; x <= hi[0]; ++x) {
                    uint8_t& cellv = g[((size_t)z*R + y)*R + x];
                    if (cellv == 1) continue;
                    const float bc[3] = { org[0] + (x + 0.5f)*cell,
                                          org[1] + (y + 0.5f)*cell,
                                          org[2] + (z + 0.5f)*cell };
                    if (tri_cell_overlap(bc, h, A, B, C)) cellv = 1;
                }
    }
}

}  // namespace

int64_t strip_interior(std::vector<float>& verts, std::vector<int32_t>& faces,
                       const StripOpts& opt) {
    const int64_t V = (int64_t)verts.size() / 3, F = (int64_t)faces.size() / 3;
    if (F == 0) return 0;

    float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
    for (int64_t v = 0; v < V; ++v)
        for (int k = 0; k < 3; ++k) {
            bmin[k] = std::min(bmin[k], verts[3*v+k]);
            bmax[k] = std::max(bmax[k], verts[3*v+k]);
        }
    float ext = 0.f;
    for (int k = 0; k < 3; ++k) ext = std::max(ext, bmax[k] - bmin[k]);
    if (ext <= 0.f) return 0;

    // Two cells of air all round, so the fill always has somewhere to start.
    const int R = std::max(32, opt.grid);
    const float cell = ext / (float)(R - 6);
    float org[3];
    for (int k = 0; k < 3; ++k) org[k] = bmin[k] - 3.0f * cell;

    const size_t N = (size_t)R * R * R;
    std::vector<uint8_t> g(N, 0);          // 0 = air, 1 = solid, 2 = exterior
    voxelize(verts, faces, org, cell, R, g);
    size_t solid = 0;
    for (size_t i = 0; i < N; ++i) solid += (g[i] == 1);
    if (opt.verbose)
        printf("  strip_interior: voxelised at %d^3 (cell %.3fmm), %zu solid cells (%.1f%%)\n",
               R, cell * 1000.0f, solid, 100.0 * (double)solid / (double)N);

    // Thicken the shell before flooding. Marking each triangle's bounding box
    // ought to be a seal on its own, but the remesh output carries ~348k
    // non-manifold edges where sheets meet, and around those the surface does
    // not cleanly separate inside from outside. Measured: at 512^3 the fill left
    // 13.4% of cells enclosed, at 1024^3 only 1.0% - the finer the cell, the
    // thinner the seal and the more the fill leaks through into the cavity.
    // One 26-neighbour dilation closes any single-cell gap. The give-back below
    // undoes it for the face test.
    for (int s = 0; s < opt.seal; ++s) {
        std::vector<int32_t> add;
        for (size_t c = 0; c < N; ++c) {
            if (g[c] != 0) continue;
            const int x = (int)(c % R), y = (int)((c / R) % R), z = (int)(c / ((size_t)R * R));
            bool near = false;
            for (int dz = -1; dz <= 1 && !near; ++dz)
                for (int dy = -1; dy <= 1 && !near; ++dy)
                    for (int dx = -1; dx <= 1 && !near; ++dx) {
                        const int ax = x+dx, ay = y+dy, az = z+dz;
                        if (ax < 0 || ay < 0 || az < 0 || ax >= R || ay >= R || az >= R) continue;
                        if (g[((size_t)az*R + ay)*R + ax] == 1) near = true;
                    }
            if (near) add.push_back((int32_t)c);
        }
        for (const int32_t c : add) g[(size_t)c] = 1;
        solid += add.size();
        if (opt.verbose)
            printf("  strip_interior: seal %d -> +%zu cells (%zu solid)\n", s + 1, add.size(), solid);
    }

    // Flood the exterior from a corner. 6-connectivity: a diagonal step could
    // squeeze between two solid cells that share only an edge and leak inside.
    //
    // Breadth-first by layers, keeping only the wavefront. A depth-first stack
    // can end up holding a large fraction of the filled volume, which at 1024^3
    // is gigabytes; a frontier is bounded by area, so it stays in the megabytes.
    if (g[0] != 0) {
        if (opt.verbose) printf("  strip_interior: corner cell is solid, aborting\n");
        return 0;
    }
    std::vector<int32_t> cur, nxt;
    g[0] = 2;
    cur.push_back(0);
    size_t exterior = 0;
    while (!cur.empty()) {
        exterior += cur.size();
        nxt.clear();
        for (const int32_t c : cur) {
            const int x = c % R, y = (c / R) % R, z = (int)((size_t)c / ((size_t)R * R));
            const int ax[6] = {x-1, x+1, x, x, x, x};
            const int ay[6] = {y, y, y-1, y+1, y, y};
            const int az[6] = {z, z, z, z, z-1, z+1};
            for (int i = 0; i < 6; ++i) {
                if (ax[i] < 0 || ay[i] < 0 || az[i] < 0 || ax[i] >= R || ay[i] >= R || az[i] >= R) continue;
                const size_t n = ((size_t)az[i]*R + ay[i])*R + ax[i];
                if (g[n] != 0) continue;
                g[n] = 2;
                nxt.push_back((int32_t)n);
            }
        }
        cur.swap(nxt);
    }
    if (opt.verbose)
        printf("  strip_interior: exterior = %zu cells (%.1f%%), enclosed air = %zu cells\n",
               exterior, 100.0 * (double)exterior / (double)N, N - exterior - solid);

    // Depth field: how many cells from the exterior is each cell? The sheets sit
    // at known separations (2*eps ~= 3.9mm, four cells at 1024^3), so a
    // histogram of per-face depth shows them as distinct peaks and the cut
    // between the outer sheet and the next one can be read off rather than
    // guessed at.
    std::vector<uint8_t> depth(N, 255);
    {
        std::vector<int32_t> cur, nxt;
        for (size_t c = 0; c < N; ++c) if (g[c] == 2) { depth[c] = 0; cur.push_back((int32_t)c); }
        uint8_t lvl = 0;
        while (!cur.empty() && lvl < 254) {
            ++lvl;
            nxt.clear();
            for (const int32_t c : cur) {
                const int x = c % R, y = (c / R) % R, z = (int)((size_t)c / ((size_t)R * R));
                const int ax[6] = {x-1, x+1, x, x, x, x};
                const int ay[6] = {y, y, y-1, y+1, y, y};
                const int az[6] = {z, z, z, z, z-1, z+1};
                for (int i = 0; i < 6; ++i) {
                    if (ax[i] < 0 || ay[i] < 0 || az[i] < 0 || ax[i] >= R || ay[i] >= R || az[i] >= R) continue;
                    const size_t n = ((size_t)az[i]*R + ay[i])*R + ax[i];
                    if (depth[n] != 255) continue;
                    depth[n] = lvl;
                    nxt.push_back((int32_t)n);
                }
            }
            cur.swap(nxt);
        }
    }

    // Per-face depth = the shallowest cell it touches.
    std::vector<uint8_t> fd((size_t)F, 255);
    const float invc = 1.0f / cell;
    for (int64_t f = 0; f < F; ++f) {
        int lo[3], hi[3];
        for (int k = 0; k < 3; ++k) {
            float a = verts[3*(size_t)faces[3*f+0]+k];
            float b = verts[3*(size_t)faces[3*f+1]+k];
            float c2 = verts[3*(size_t)faces[3*f+2]+k];
            const float mn = std::min(a, std::min(b, c2)), mx = std::max(a, std::max(b, c2));
            lo[k] = std::max(0, std::min(R-1, (int)std::floor((mn - org[k]) * invc)));
            hi[k] = std::max(0, std::min(R-1, (int)std::floor((mx - org[k]) * invc)));
        }
        uint8_t best = 255;
        for (int z = lo[2]; z <= hi[2]; ++z)
            for (int y = lo[1]; y <= hi[1]; ++y)
                for (int x = lo[0]; x <= hi[0]; ++x)
                    best = std::min(best, depth[((size_t)z*R + y)*R + x]);
        fd[(size_t)f] = best;
    }
    int64_t hist[20] = {0};
    for (int64_t f = 0; f < F; ++f) hist[std::min<int>(fd[(size_t)f], 19)] += 1;
    if (opt.verbose) {
        printf("  strip_interior: faces by depth from the exterior (cells):\n");
        int64_t run = 0;
        for (int d = 0; d < 20; ++d) {
            if (!hist[d]) continue;
            run += hist[d];
            printf("  strip_interior:    depth %-3d %10lld  (cumulative %.1f%%)\n",
                   d, (long long)hist[d], 100.0 * (double)run / (double)F);
        }
    }

    std::vector<uint8_t> vis((size_t)F, 0);
    int64_t seen = 0;

    // ---- exterior-side classification (opt.outward) ---------------------
    // Keep a face when the space on its front side is connected to infinity.
    //
    // That question is global, which is why no local test can answer it: an
    // inner-wall face has empty space ahead of it too (the object's own
    // interior), so locally it is indistinguishable from an outer-wall face.
    // Only connectivity separates them, and connectivity is the flood fill.
    //
    // The flood must therefore not leak -- and at the mesh's own scale it does.
    // Sealing this shell takes eight cells of dilation (measured: enclosed air
    // 1 -> 62,359,268 between seal 4 and 8) while resolving its two walls, four
    // cells apart, needs fewer than four. Both requirements scale with the cell,
    // so no single resolution satisfies them.
    //
    // Run this classification at a COARSE grid instead. At a cell comparable to
    // the shell thickness the shell is one solid cell thick and seals with no
    // dilation at all, while the object's interior is many cells across and
    // floods as a distinct region. Resolution is then no longer being asked to
    // separate the two walls -- the probe direction does that.
    if (opt.outward) {
        auto is_ext = [&](const float p3[3]) -> bool {
            int c3[3];
            for (int k = 0; k < 3; ++k) {
                c3[k] = (int)std::floor((p3[k] - org[k]) * invc);
                if (c3[k] < 0 || c3[k] >= R) return true;   // beyond the grid is outside
            }
            return g[((size_t)c3[2]*R + c3[1])*R + c3[0]] == 2;
        };
        // Far enough to clear the shell, near enough to stay inside the cavity
        // rather than punching out the other side of a thin feature.
        const float d1 = 1.5f * cell;

        auto face_geom = [&](int64_t f, float ctr[3], float nrm[3]) {
            const float* A = &verts[3*(size_t)faces[3*f+0]];
            const float* B = &verts[3*(size_t)faces[3*f+1]];
            const float* C = &verts[3*(size_t)faces[3*f+2]];
            for (int k = 0; k < 3; ++k) ctr[k] = (A[k] + B[k] + C[k]) / 3.f;
            const float e1[3] = {B[0]-A[0], B[1]-A[1], B[2]-A[2]};
            const float e2[3] = {C[0]-A[0], C[1]-A[1], C[2]-A[2]};
            nrm[0] = e1[1]*e2[2]-e1[2]*e2[1];
            nrm[1] = e1[2]*e2[0]-e1[0]*e2[2];
            nrm[2] = e1[0]*e2[1]-e1[1]*e2[0];
            const float l = std::sqrt(nrm[0]*nrm[0]+nrm[1]*nrm[1]+nrm[2]*nrm[2]);
            if (l > 1e-20f) { for (int k = 0; k < 3; ++k) nrm[k] /= l; }
            else { nrm[0] = 0.f; nrm[1] = 0.f; nrm[2] = 1.f; }
        };

        // clean_mesh unifies winding but has no notion of outward, so vote on it:
        // with the correct sign the outer wall sees exterior ahead, with it
        // flipped almost nothing does.
        int64_t votes[2] = {0, 0};
        const int64_t stride = F > 20000 ? F / 20000 : 1;
        int64_t sampled = 0;
        for (int64_t f = 0; f < F; f += stride) {
            ++sampled;
            float ctr[3], nrm[3];
            face_geom(f, ctr, nrm);
            for (int sg = 0; sg < 2; ++sg) {
                const float m = sg ? -1.f : 1.f;
                const float p1[3] = { ctr[0]+m*nrm[0]*d1, ctr[1]+m*nrm[1]*d1, ctr[2]+m*nrm[2]*d1 };
                if (is_ext(p1)) ++votes[sg];
            }
        }
        const float sgn = votes[1] > votes[0] ? -1.f : 1.f;
        if (opt.verbose)
            printf("  strip_interior: outward sign %+.0f (exterior ahead: +%lld / -%lld of %lld)\n",
                   sgn, (long long)votes[0], (long long)votes[1], (long long)sampled);

        for (int64_t f = 0; f < F; ++f) {
            float ctr[3], nrm[3];
            face_geom(f, ctr, nrm);
            const float p1[3] = { ctr[0]+sgn*nrm[0]*d1, ctr[1]+sgn*nrm[1]*d1, ctr[2]+sgn*nrm[2]*d1 };
            if (is_ext(p1)) { vis[(size_t)f] = 1; ++seen; }
        }
        if (opt.verbose)
            printf("  strip_interior: exterior-side test keeps %lld/%lld faces (%.1f%%)\n",
                   (long long)seen, (long long)F, 100.0 * (double)seen / (double)F);
        if (seen == 0 || seen == F) {
            if (opt.verbose) printf("  strip_interior: nothing to remove\n");
            return 0;
        }
    } else {

    // Where to cut. The outer sheet and whatever lies under it show up as two
    // populations separated by a near-empty trough - the 2*eps gap, in cells.
    // Finding that trough per asset matters: measured troughs sit at depth 5 on
    // a teddy bear and an alarm clock but at 6 on a hat, and a table and a tower
    // have no second population at all (their histograms just decay), in which
    // case there is nothing buried and the mesh must be left alone. A fixed
    // depth would silently keep both sheets on one asset and cut into the skin
    // on another.
    int cut = opt.depth;
    if (cut < 0) {
        int64_t peak = 0;
        for (int d = 2; d < 19; ++d) peak = std::max(peak, hist[d]);
        for (int d = 3; d < 18; ++d) {
            if (hist[d] < hist[d-1] && hist[d+1] > 2 * hist[d] && hist[d] * 5 < peak) {
                cut = d;
                break;
            }
        }
        if (cut < 0) {
            if (opt.verbose)
                printf("  strip_interior: no trough in the depth histogram - nothing is buried, "
                       "leaving all %lld faces\n", (long long)F);
            return 0;
        }
        if (opt.verbose)
            printf("  strip_interior: trough at depth %d (%lld faces, peak %lld) -> cutting there\n",
                   cut, (long long)hist[cut], (long long)peak);
    }

    // Keep everything down to the chosen depth.
    for (int64_t f = 0; f < F; ++f)
        if (fd[(size_t)f] <= cut) { vis[(size_t)f] = 1; ++seen; }
    if (opt.verbose)
        printf("  strip_interior: keeping depth <= %d -> %lld/%lld faces (%.1f%%)\n",
               cut, (long long)seen, (long long)F, 100.0 * (double)seen / (double)F);
    if (seen == 0) return 0;
    }   // end of the trough path

    // Compact.
    std::vector<int32_t> remap((size_t)V, -1);
    std::vector<float> nv;
    std::vector<int32_t> nf;
    nf.reserve((size_t)seen * 3);
    for (int64_t f = 0; f < F; ++f) {
        if (!vis[(size_t)f]) continue;
        for (int j = 0; j < 3; ++j) {
            const int32_t o = faces[3*f+j];
            int32_t& m = remap[(size_t)o];
            if (m < 0) {
                m = (int32_t)(nv.size() / 3);
                nv.push_back(verts[3*(size_t)o+0]);
                nv.push_back(verts[3*(size_t)o+1]);
                nv.push_back(verts[3*(size_t)o+2]);
            }
            nf.push_back(m);
        }
    }
    const int64_t removed = F - (int64_t)(nf.size() / 3);
    if (opt.verbose)
        printf("  strip_interior: V %lld->%lld, F %lld->%lld (removed %lld, %.1f%%)\n",
               (long long)V, (long long)(nv.size()/3), (long long)F,
               (long long)(nf.size()/3), (long long)removed,
               100.0 * (double)removed / (double)F);
    verts.swap(nv);
    faces.swap(nf);
    return removed;
}

}  // namespace trellis
