// Write to a sidecar and rename into place, so a partial file is never visible
// under the real name.
//
// This matters for two consumers, both real:
//
//   * The resume cache and the master. A truncated one read back as a cache HIT
//     is worse than a miss -- it is a wrong answer that looks like a right one.
//
//   * The output GLB. The web front end treats the .glb appearing on disk as the
//     marker that an asset's generation FINISHED (job_progress() in
//     a calling application), so a half-written one is reported as a completed asset.
//
// And the front end kills jobs with `killing the process tree`, which can land anywhere --
// including the middle of a 10 MB write.
#pragma once
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

namespace trellis {

inline std::string part_path(const std::string& path) { return path + ".part"; }

// Rename "<path>.part" over `path`. std::filesystem::rename replaces an existing
// destination on both platforms (MoveFileEx with MOVEFILE_REPLACE_EXISTING on
// Windows), which plain rename(3) does not do there.
inline bool commit_part(const std::string& path) {
    std::error_code ec;
    std::filesystem::rename(part_path(path), path, ec);
    if (ec) {
        fprintf(stderr, "      [atomic] cannot commit %s: %s\n",
                path.c_str(), ec.message().c_str());
        std::error_code rm;
        std::filesystem::remove(part_path(path), rm);
        return false;
    }
    return true;
}

// Drop a partial file after a failed write, so the next run does not trip over it.
inline void discard_part(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(part_path(path), ec);
}

}  // namespace trellis
