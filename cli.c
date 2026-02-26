#include <stdio.h>
#include <stdlib.h>
#include "tq.h"

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: tq <file.fvecs>\n"); return 1; }
    int n, d;
    float *vecs = load_fvecs(argv[1], &n, &d);
    if (!vecs) return 1;
    printf("loaded %d vectors d=%d\n", n, d);
    float dist = l2dist(vecs, vecs + d, d);
    printf("l2dist(v0, v1) = %f\n", dist);
    free(vecs);
    return 0;
}
