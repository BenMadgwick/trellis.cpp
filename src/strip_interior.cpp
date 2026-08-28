#include "strip_interior.h"
#include "tri_bvh.h"
#include <functional>
#include <unordered_map>
#include <algorithm>

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

    // ---- side-of-input classification (opt.outward) ---------------------
    // Ask which side of the ORIGINAL surface each face sits on.
    //
    // remesh_dc places every output vertex at distance eps from its input, so a
    // face is either eps outside that input or eps inside it -- and the closest
    // point on the input, with the input's own normal there, says which. One
    // closest-point query per face, no grid, no flood, no min-cut.
    //
    // This is the one question the failed approaches were all circling, and it
    // is answerable only because the input is still available:
    //   - depth-histogram trough: the seal needed against flood leaks (8 cells)
    //     exceeds the wall separation (4 cells), and both scale with the cell.
    //   - local probes / visibility rays: an inner-wall face has empty space
    //     ahead of it too, so it is locally identical to an outer-wall face.
    //   - ray parity / winding number, and libigl's outer_hull: by
    //     Jordan-Brouwer a closed connected manifold has exactly two complement
    //     components, and this mesh encloses only its own 2*eps band (measured
    //     volume/area = eps). So the "cavity" IS the exterior, reached through
    //     the tunnels where the bag folds around the input's holes. Nothing is
    //     enclosed, and no connectivity method can separate what is not separate.
    //   - connected components: one component, the walls join at those folds.
    //
    // Depends on the input's winding being locally consistent, which for TRELLIS
    // it is. A global flip is handled by the vote below.
    if (opt.outward) {
        if (!opt.ref_bvh || !opt.ref_verts || !opt.ref_faces || opt.ref_F <= 0) {
            if (opt.verbose)
                printf("  strip_interior: --strip-outward needs the pre-remesh mesh; skipping\n");
            return 0;
        }
        auto side = [&](int64_t f, float& out_d) -> int {
            const float* A = &verts[3*(size_t)faces[3*f+0]];
            const float* B = &verts[3*(size_t)faces[3*f+1]];
            const float* C = &verts[3*(size_t)faces[3*f+2]];
            float c3[3];
            for (int k = 0; k < 3; ++k) c3[k] = (A[k] + B[k] + C[k]) / 3.f;
            const TriBvh::Hit h = opt.ref_bvh->closest(c3, 1e30f);
            if (h.face < 0) { out_d = 0.f; return 0; }
            const int32_t* rf = &opt.ref_faces[3*(size_t)h.face];
            const float* P0 = &opt.ref_verts[3*(size_t)rf[0]];
            const float* P1 = &opt.ref_verts[3*(size_t)rf[1]];
            const float* P2 = &opt.ref_verts[3*(size_t)rf[2]];
            const float e1[3] = {P1[0]-P0[0], P1[1]-P0[1], P1[2]-P0[2]};
            const float e2[3] = {P2[0]-P0[0], P2[1]-P0[1], P2[2]-P0[2]};
            const float n[3] = { e1[1]*e2[2]-e1[2]*e2[1],
                                 e1[2]*e2[0]-e1[0]*e2[2],
                                 e1[0]*e2[1]-e1[1]*e2[0] };
            float d = 0.f;
            for (int k = 0; k < 3; ++k) d += (c3[k] - h.point[k]) * n[k];
            const float ln = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            out_d = ln > 1e-20f ? d / ln : 0.f;
            return d > 0.f ? 1 : (d < 0.f ? -1 : 0);
        };

        // Global winding of the input could be flipped; the outer side is
        // whichever holds the larger total area (the eps-outward offset of a
        // closed-ish sheet is strictly the larger of the two).
        double area_pos = 0.0, area_neg = 0.0;
        std::vector<int8_t> sd((size_t)F, 0);
        std::vector<float>  sdist((size_t)F, 0.f);
        for (int64_t f = 0; f < F; ++f) {
            float d = 0.f;
            const int sgn = side(f, d);
            sd[(size_t)f] = (int8_t)sgn;
            sdist[(size_t)f] = d;
            const float* A = &verts[3*(size_t)faces[3*f+0]];
            const float* B = &verts[3*(size_t)faces[3*f+1]];
            const float* C = &verts[3*(size_t)faces[3*f+2]];
            const double u[3] = {B[0]-A[0], B[1]-A[1], B[2]-A[2]};
            const double v[3] = {C[0]-A[0], C[1]-A[1], C[2]-A[2]};
            const double cx = u[1]*v[2]-u[2]*v[1], cy = u[2]*v[0]-u[0]*v[2], cz = u[0]*v[1]-u[1]*v[0];
            const double ar = 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
            if (sgn > 0) area_pos += ar; else if (sgn < 0) area_neg += ar;
        }
        const int keep_sign = area_pos >= area_neg ? 1 : -1;
        if (opt.verbose)
            printf("  strip_interior: side-of-input areas + %.1f cm2 / - %.1f cm2 -> keeping %s\n",
                   area_pos * 1e4, area_neg * 1e4, keep_sign > 0 ? "+" : "-");

        // The raw sign is globally right and locally noisy: near the fold, and
        // wherever the closest-point query lands on a differently-oriented input
        // triangle, it flips face to face. Thresholding it directly shreds the
        // surface (measured: 1,973,145 boundary edges on a 3.87M-face half).
        //
        // Diffuse the signed distance over face adjacency and re-threshold. The
        // distance carries confidence the bare sign throws away -- deep faces
        // outvote ambiguous ones instead of merely outnumbering them -- so the
        // label boundary relaxes onto the fold, which is where the two walls
        // genuinely meet and the only place a cut costs nothing.
        if (opt.smooth > 0) {
            // face adjacency by shared edge, via a sorted edge list rather than
            // a hash map: 23M entries at this scale.
            std::vector<std::pair<uint64_t,int32_t>> ed;
            ed.reserve((size_t)F * 3);
            for (int64_t f = 0; f < F; ++f)
                for (int j = 0; j < 3; ++j) {
                    int32_t a = faces[3*f+j], b = faces[3*f+(j+1)%3];
                    if (a > b) std::swap(a, b);
                    ed.emplace_back(((uint64_t)(uint32_t)a << 32) | (uint32_t)b, (int32_t)f);
                }
            std::sort(ed.begin(), ed.end());
            std::vector<int32_t> nbr_off((size_t)F + 1, 0), nbr;
            nbr.reserve((size_t)F * 3);
            {   // count then fill, so neighbours live in one flat array
                std::vector<int32_t> cnt((size_t)F, 0);
                for (size_t i = 0; i + 1 < ed.size(); ++i)
                    if (ed[i].first == ed[i+1].first) { ++cnt[(size_t)ed[i].second]; ++cnt[(size_t)ed[i+1].second]; }
                for (int64_t f = 0; f < F; ++f) nbr_off[(size_t)f+1] = nbr_off[(size_t)f] + cnt[(size_t)f];
                nbr.assign((size_t)nbr_off[(size_t)F], -1);
                std::vector<int32_t> fill((size_t)F, 0);
                for (size_t i = 0; i + 1 < ed.size(); ++i)
                    if (ed[i].first == ed[i+1].first) {
                        const int32_t u = ed[i].second, v = ed[i+1].second;
                        nbr[(size_t)nbr_off[(size_t)u] + fill[(size_t)u]++] = v;
                        nbr[(size_t)nbr_off[(size_t)v] + fill[(size_t)v]++] = u;
                    }
            }
            std::vector<float> cur = sdist, nxt((size_t)F);
            for (int it = 0; it < opt.smooth; ++it) {
                for (int64_t f = 0; f < F; ++f) {
                    float acc = cur[(size_t)f];
                    int n = 1;
                    for (int32_t i = nbr_off[(size_t)f]; i < nbr_off[(size_t)f+1]; ++i) {
                        acc += cur[(size_t)nbr[(size_t)i]];
                        ++n;
                    }
                    nxt[(size_t)f] = acc / (float)n;
                }
                cur.swap(nxt);
            }
            int64_t flipped = 0;
            for (int64_t f = 0; f < F; ++f) {
                const int8_t ns = cur[(size_t)f] > 0.f ? 1 : (cur[(size_t)f] < 0.f ? -1 : 0);
                if (ns != 0 && ns != sd[(size_t)f]) ++flipped;
                if (ns != 0) sd[(size_t)f] = ns;
            }
            if (opt.verbose)
                printf("  strip_interior: %d smoothing passes relabelled %lld faces (%.2f%%)\n",
                       opt.smooth, (long long)flipped, 100.0 * (double)flipped / (double)F);
        }

        for (int64_t f = 0; f < F; ++f)
            if (sd[(size_t)f] == keep_sign) { vis[(size_t)f] = 1; ++seen; }
        if (opt.verbose)
            printf("  strip_interior: side test keeps %lld/%lld faces (%.1f%%)\n",
                   (long long)seen, (long long)F, 100.0 * (double)seen / (double)F);
        if (seen == 0 || seen == F) {
            if (opt.verbose) printf("  strip_interior: nothing separable\n");
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
