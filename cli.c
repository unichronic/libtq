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

    float *q = vecs;
    float tot_err = 0, tot_true = 0;
    int check = n < 1000 ? n : 1000;
    for (int i = 0; i < check; i++) {
        float true_dot = 0;
        for (int j = 0; j < d; j++) true_dot += q[j]*vecs[i*d+j];
        tq_vec cv = tq_compress(&ctx, vecs+i*d);
        float est = tq_dot(&ctx, q, &cv);
        tot_err  += fabsf(true_dot-est);
        tot_true += fabsf(true_dot);
        tq_vec_free(&cv);
    }
    printf("dot error rel=%.3f\n", tot_err/tot_true);

    tq_free(&ctx);
    free(vecs);
    return 0;
}
