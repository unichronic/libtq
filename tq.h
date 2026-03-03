#ifndef TQ_H
#define TQ_H

#include <stdint.h>

typedef struct {
    int    nlvl;
    float *cents;
    float *bounds;
} tq_cb;

typedef struct {
    int   d;
    int   bits;
    tq_cb cb;
} tq_ctx;

float  l2dist(float *a, float *b, int d);
float *load_fvecs(const char *path, int *n, int *d);
void   solve_lloyd_max(int d, int bits, float *cents, float *bounds);
int    tq_init(tq_ctx *ctx, int d, int bits);
void   tq_free(tq_ctx *ctx);

#endif
