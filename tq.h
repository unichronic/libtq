#ifndef TQ_H
#define TQ_H

#include <stdint.h>

typedef struct {
    int    nlvl;
    float *cents;
    float *bounds;
} tq_cb;

typedef struct {
    uint8_t *idx;
    float    rnorm;
    int      d;
    int      bits;
} tq_vec;

typedef struct {
    int    d;
    int    bits;
    float *rot;
    tq_cb  cb;
} tq_ctx;

float  l2dist(float *a, float *b, int d);
float *load_fvecs(const char *path, int *n, int *d);
int    tq_init(tq_ctx *ctx, int d, int bits, unsigned int seed);
void   tq_free(tq_ctx *ctx);
tq_vec tq_compress(tq_ctx *ctx, float *v);
void   tq_decompress(tq_ctx *ctx, tq_vec *cv, float *out);
void   tq_vec_free(tq_vec *cv);

#endif
