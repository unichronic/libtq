#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tq.h"

#define D 128
#define N 500

static float rng01(unsigned int *s) {
    *s = *s * 1664525u + 1013904223u;
    return (*s & 0x7fffffff) / 2147483648.0f;
}

static float randn(unsigned int *s) {
    float u1 = rng01(s) + 1e-9f;
    float u2 = rng01(s);
    return sqrtf(-2*logf(u1)) * cosf(2*3.14159265f*u2);
}

int main(void) {
    int pass = 0, fail = 0;

    float vecs[N][D];
    unsigned int seed = 1234;
    for (int i = 0; i < N; i++) {
        float norm = 0;
        for (int j = 0; j < D; j++) { vecs[i][j] = randn(&seed); norm += vecs[i][j]*vecs[i][j]; }
        norm = sqrtf(norm);
        for (int j = 0; j < D; j++) vecs[i][j] /= norm;
    }

    int bits_list[] = {2, 3, 4};
    for (int bi = 0; bi < 3; bi++) {
        int bits = bits_list[bi];
        tq_ctx ctx;
        tq_init(&ctx, D, bits, 42);

        float bias_sum = 0;
        int npairs = 200;
        for (int i = 0; i < npairs; i++) {
            float *q = vecs[i];
            float *k = vecs[npairs + i];
            float true_dot = 0;
            for (int j = 0; j < D; j++) true_dot += q[j]*k[j];
            tq_vec cv = tq_compress(&ctx, k);
            float est = tq_dot(&ctx, q, &cv);
            bias_sum += est - true_dot;
            tq_vec_free(&cv);
        }
        float bias = bias_sum / npairs;
        int ok = fabsf(bias) < 0.05f;
        printf("[%s] bits=%d  dot bias=%.4f\n", ok?"PASS":"FAIL", bits, bias);
        if (ok) pass++; else fail++;

        float *q = vecs[0];
        tq_vec cvs[N];
        for (int i = 0; i < N; i++) cvs[i] = tq_compress(&ctx, vecs[i]);

        float best = -1e9f; int best_i = -1;
        for (int i = 0; i < N; i++) {
            float s = tq_dot(&ctx, q, &cvs[i]);
            if (s > best) { best = s; best_i = i; }
        }
        float tbest = -1e9f; int tbest_i = -1;
        for (int i = 0; i < N; i++) {
            float s = 0;
            for (int j = 0; j < D; j++) s += q[j]*vecs[i][j];
            if (s > tbest) { tbest = s; tbest_i = i; }
        }
        int match = (best_i == tbest_i);
        printf("[%s] bits=%d  top1: tq=%d true=%d\n", match?"PASS":"FAIL", bits, best_i, tbest_i);
        if (match) pass++; else fail++;

        for (int i = 0; i < N; i++) tq_vec_free(&cvs[i]);
        tq_free(&ctx);
    }

    printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
