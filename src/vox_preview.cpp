#include "vox_preview.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_set>

// Declarations only; STB_IMAGE_WRITE_IMPLEMENTATION lives in mesh_glb.cpp.
#include "stb_image_write.h"

namespace trellis {

namespace {

struct Vec3 { float x, y, z; };

inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

// Voxel key for the occupancy set, so interior faces can be skipped. At 32^3 a
// 24%-occupied volume hides most of its own faces; drawing them would cost more
// and buy nothing but z-fighting.
inline uint32_t vkey(int x, int y, int z) {
    return ((uint32_t)(x & 1023) << 20) | ((uint32_t)(y & 1023) << 10) | (uint32_t)(z & 1023);
}

// Yaw about Y then pitch about X. Right-handed, -Z into the screen.
struct View {
    const char* name;
    float yaw_deg, pitch_deg;
};

Vec3 rotate(const Vec3& p, float cy, float sy, float cp, float sp) {
    const float x =  p.x * cy + p.z * sy;
    const float z = -p.x * sy + p.z * cy;
    const float y =  p.y * cp - z * sp;
    const float z2 = p.y * sp + z * cp;
    return { x, y, z2 };
}

// The 6 cube faces: 4 corner offsets (unit cube centred on the origin) + normal.
const int FACE[6][4][3] = {
    {{ 1,-1,-1},{ 1, 1,-1},{ 1, 1, 1},{ 1,-1, 1}},   // +x
    {{-1,-1, 1},{-1, 1, 1},{-1, 1,-1},{-1,-1,-1}},   // -x
    {{-1, 1,-1},{ 1, 1,-1},{ 1, 1, 1},{-1, 1, 1}},   // +y
    {{-1,-1, 1},{ 1,-1, 1},{ 1,-1,-1},{-1,-1,-1}},   // -y
    {{-1,-1, 1},{-1, 1, 1},{ 1, 1, 1},{ 1,-1, 1}},   // +z
    {{ 1,-1,-1},{ 1, 1,-1},{-1, 1,-1},{-1,-1,-1}},   // -z
};
const int FACE_N[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};

struct Raster {
    int w = 0, h = 0;
    std::vector<float> depth;
    std::vector<uint8_t> rgb;
    void init(int W, int H, uint8_t bg) {
        w = W; h = H;
        depth.assign((size_t)W*H, 1e30f);
        rgb.assign((size_t)W*H*3, bg);
    }
    void tri(const float p0[3], const float p1[3], const float p2[3], const uint8_t c[3]) {
        const float minxf = std::min({p0[0], p1[0], p2[0]});
        const float maxxf = std::max({p0[0], p1[0], p2[0]});
        const float minyf = std::min({p0[1], p1[1], p2[1]});
        const float maxyf = std::max({p0[1], p1[1], p2[1]});
        int x0 = std::max(0, (int)std::floor(minxf)), x1 = std::min(w-1, (int)std::ceil(maxxf));
        int y0 = std::max(0, (int)std::floor(minyf)), y1 = std::min(h-1, (int)std::ceil(maxyf));
        if (x0 > x1 || y0 > y1) return;
        const float ax = p1[0]-p0[0], ay = p1[1]-p0[1];
        const float bx = p2[0]-p0[0], by = p2[1]-p0[1];
        const float den = ax*by - bx*ay;
        if (std::fabs(den) < 1e-12f) return;
        const float inv = 1.0f / den;
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const float px = x + 0.5f - p0[0], py = y + 0.5f - p0[1];
                const float u = (px*by - bx*py) * inv;
                const float v = (ax*py - px*ay) * inv;
                if (u < 0.f || v < 0.f || u + v > 1.f) continue;
                const float d = p0[2] + u*(p1[2]-p0[2]) + v*(p2[2]-p0[2]);
                const size_t i = (size_t)y*w + x;
                if (d >= depth[i]) continue;
                depth[i] = d;
                rgb[i*3+0] = c[0]; rgb[i*3+1] = c[1]; rgb[i*3+2] = c[2];
            }
    }
};

}  // namespace

bool write_vox_preview(const std::vector<std::array<int,3>>& coords, int grid,
                       const std::string& path, int tile) {
    if (coords.empty() || grid <= 0 || tile < 32) return false;

    std::unordered_set<uint32_t> occ;
    occ.reserve(coords.size() * 2);
    for (const auto& c : coords) occ.insert(vkey(c[0], c[1], c[2]));

    // World placement: voxel i spans [i/grid - 0.5, (i+1)/grid - 0.5], matching
    // the --voxply dump so the preview and that point cloud agree.
    const float vs = 1.0f / (float)grid;          // voxel size
    const float hs = 0.5f * vs;                   // half size

    // Centre on the OCCUPIED bounding box, and share one scale across all four
    // views so the tiles are directly comparable rather than each self-fitted.
    int lo[3] = {grid, grid, grid}, hi[3] = {-1, -1, -1};
    for (const auto& c : coords)
        for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], c[k]); hi[k] = std::max(hi[k], c[k]); }
    const Vec3 centre = { ((lo[0]+hi[0])*0.5f + 0.5f)*vs - 0.5f,
                          ((lo[1]+hi[1])*0.5f + 0.5f)*vs - 0.5f,
                          ((lo[2]+hi[2])*0.5f + 0.5f)*vs - 0.5f };
    float extent = 0.f;
    for (int k = 0; k < 3; ++k) extent = std::max(extent, (hi[k]-lo[k]+1) * vs);
    // Room for the diagonal so the three-quarter view is not clipped.
    const float half = extent * 0.72f;
    const float scale = (tile * 0.5f) / half;

    // A full turntable at 45 deg intervals, every view pitched 20 deg down so the
    // top face reads as well as the sides. A straight-on view (pitch 0) shows a
    // silhouette and nothing of the upper surface, which is where a wrong
    // generation usually gives itself away.
    static const View views[8] = {
        { "000", 0.0f, 20.0f }, { "045",  45.0f, 20.0f },
        { "090", 90.0f, 20.0f }, { "135", 135.0f, 20.0f },
        { "180", 180.0f, 20.0f }, { "225", 225.0f, 20.0f },
        { "270", 270.0f, 20.0f }, { "315", 315.0f, 20.0f },
    };
    const int NV = 8, COLS = 4;

    const int W = tile*COLS, H = tile*(NV/COLS);
    std::vector<uint8_t> out((size_t)W*H*3, 24);

    // Fixed key light, slightly off-axis so the three visible faces of a cube
    // read as three different tones.
    Vec3 L = { 0.40f, 0.78f, 0.48f };
    { const float l = std::sqrt(dot(L,L)); L.x/=l; L.y/=l; L.z/=l; }

    for (int vi = 0; vi < NV; ++vi) {
        const float yr = views[vi].yaw_deg   * 3.14159265f / 180.f;
        const float pr = views[vi].pitch_deg * 3.14159265f / 180.f;
        const float cy = std::cos(yr), sy = std::sin(yr);
        const float cp = std::cos(pr), sp = std::sin(pr);

        Raster r;
        r.init(tile, tile, 24);

        for (const auto& c : coords) {
            const Vec3 vc = { (c[0]+0.5f)*vs - 0.5f, (c[1]+0.5f)*vs - 0.5f, (c[2]+0.5f)*vs - 0.5f };
            for (int f = 0; f < 6; ++f) {
                // Skip a face whose neighbour is solid: it can never be seen.
                const int nx = c[0]+FACE_N[f][0], ny = c[1]+FACE_N[f][1], nz = c[2]+FACE_N[f][2];
                if (nx >= 0 && ny >= 0 && nz >= 0 && nx < grid && ny < grid && nz < grid &&
                    occ.count(vkey(nx, ny, nz))) continue;

                const Vec3 n = { (float)FACE_N[f][0], (float)FACE_N[f][1], (float)FACE_N[f][2] };
                const Vec3 nv = rotate(n, cy, sy, cp, sp);
                if (nv.z <= 0.0f) continue;                 // back-facing after rotation

                // Ambient occlusion from the 8 cells ringing this face's outward
                // neighbour. Without it an axis-aligned view of a boxy subject is
                // a single flat tone -- every face shares one normal, so the
                // lambert term alone carries no shape at all. AO is what makes
                // the front/side tiles show relief rather than a silhouette.
                int occl = 0;
                for (int du = -1; du <= 1; ++du)
                    for (int dv = -1; dv <= 1; ++dv) {
                        if (!du && !dv) continue;
                        // Step in the two axes perpendicular to the face normal.
                        const int a0 = (f / 2 + 1) % 3, a1 = (f / 2 + 2) % 3;
                        int q[3] = { c[0]+FACE_N[f][0], c[1]+FACE_N[f][1], c[2]+FACE_N[f][2] };
                        q[a0] += du; q[a1] += dv;
                        if (q[0] < 0 || q[1] < 0 || q[2] < 0 ||
                            q[0] >= grid || q[1] >= grid || q[2] >= grid) continue;
                        if (occ.count(vkey(q[0], q[1], q[2]))) ++occl;
                    }
                const float ao = 1.0f - 0.45f * ((float)occl / 8.0f);

                const float lam = std::max(0.0f, dot(n, L));
                const float ints = (0.28f + 0.72f * lam) * ao;
                const uint8_t col[3] = {
                    (uint8_t)std::min(255.f, 150.f * ints + 24.f),
                    (uint8_t)std::min(255.f, 168.f * ints + 26.f),
                    (uint8_t)std::min(255.f, 196.f * ints + 30.f) };

                float scr[4][3];
                for (int k = 0; k < 4; ++k) {
                    const Vec3 wp = { vc.x + FACE[f][k][0]*hs,
                                      vc.y + FACE[f][k][1]*hs,
                                      vc.z + FACE[f][k][2]*hs };
                    const Vec3 rp = rotate(wp - centre, cy, sy, cp, sp);
                    scr[k][0] = tile*0.5f + rp.x * scale;
                    scr[k][1] = tile*0.5f - rp.y * scale;     // screen y grows downward
                    scr[k][2] = -rp.z;                        // smaller = nearer
                }
                r.tri(scr[0], scr[1], scr[2], col);
                r.tri(scr[0], scr[2], scr[3], col);
            }
        }

        const int ox = (vi % COLS) * tile, oy = (vi / COLS) * tile;
        for (int y = 0; y < tile; ++y)
            std::memcpy(&out[(((size_t)(oy+y))*W + ox)*3], &r.rgb[(size_t)y*tile*3], (size_t)tile*3);
    }

    // Hairline separators between the tiles.
    for (int c = 1; c < COLS; ++c)
        for (int y = 0; y < H; ++y) { const size_t i = ((size_t)y*W + (size_t)c*tile)*3; out[i]=out[i+1]=out[i+2]=64; }
    for (int r = 1; r < NV/COLS; ++r)
        for (int x = 0; x < W; ++x) { const size_t i = ((size_t)r*tile*W + x)*3; out[i]=out[i+1]=out[i+2]=64; }

    if (!stbi_write_png(path.c_str(), W, H, 3, out.data(), W*3)) {
        fprintf(stderr, "      [vox] cannot write %s\n", path.c_str());
        return false;
    }
    printf("      [vox] preview -> %s (%dx%d, %d-view turntable at %d deg, pitched %.0f deg down)\n",
           path.c_str(), W, H, NV, 360 / NV, views[0].pitch_deg);
    fflush(stdout);
    return true;
}

}  // namespace trellis
