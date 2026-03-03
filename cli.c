#include <stdio.h>
#include <stdlib.h>
#include "tq.h"

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: tq <file.fvecs>\n"); return 1; }
    int n, d;
    float *vecs = load_fvecs(argv[1], &n, &d);
    if (!vecs) return 1;
    printf("loaded %d vectors d=%d\n", n, d);

    tq_ctx ctx;
    tq_init(&ctx, d, 3);
    printf("codebook: %d levels\n", ctx.cb.nlvl);
    printf("centroid[0]=%.4f  centroid[last]=%.4f\n",
           ctx.cb.cents[0], ctx.cb.cents[ctx.cb.nlvl-1]);
    tq_free(&ctx);
    free(vecs);
    return 0;
}
