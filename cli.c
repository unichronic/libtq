#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tq.h"

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: tq <file.fvecs>\n"); return 1; }
    int n, d;
    float *vecs = load_fvecs(argv[1], &n, &d);
    if (!vecs) return 1;
    printf("loaded %d vectors d=%d\n", n, d);

    tq_ctx ctx;
    tq_init(&ctx, d, 3, 42);

    float *recon = malloc(d*sizeof(float));
    tq_vec cv = tq_compress(&ctx, vecs);
    tq_decompress(&ctx, &cv, recon);

    float err = 0;
    for (int i = 0; i < d; i++) { float r = vecs[i]-recon[i]; err += r*r; }
    printf("recon mse on v0: %f\n", err/d);

    tq_vec_free(&cv);
    free(recon);
    tq_free(&ctx);
    free(vecs);
    return 0;
}
