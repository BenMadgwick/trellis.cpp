// Thin CLI entry point: parse args, then run the shared trellis_run() pipeline.
#include "trellis_args.h"
#include "trellis_run.h"
#include <cstdio>

int main(int argc, char** argv) {
    trellis::TrellisParams p;
    if (!trellis::parse_args(argc, argv, p)) {
        trellis::print_usage(argv[0], /*server=*/false);
        return p.help ? 0 : 1;
    }
    // --load-vox resumes after DINOv3, and the cache carries the conditioning,
    // so a resumed run genuinely has no use for the image.
    if (p.image.empty() && p.load_vox.empty()) {
        fprintf(stderr, "[trellis] no input image (give <image.png>, --image, or --load-vox CACHE)\n");
        trellis::print_usage(argv[0], /*server=*/false);
        return 1;
    }
    if (!p.image.empty() && !p.load_vox.empty())
        fprintf(stderr, "[trellis] --load-vox given: ignoring the input image "
                        "(the cache already holds its conditioning)\n");
    return trellis_run(p);
}
