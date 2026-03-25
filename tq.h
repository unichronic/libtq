#ifndef TQ_H
#define TQ_H

#include <stdint.h>

typedef struct {
    int    nlvl;
    float *cents;
    float *bounds;
} tq_cb;

// ibuf = quantized indices packed tightly, sbuf = sign bits from qjl (1 bit each)
// rnorm16 = residual norm stored as fp16 to save space
typedef struct {
    uint8_t *ibuf;
    uint8_t *sbuf;
    uint16_t rnorm16;
    int      d;
    int      bits;
} tq_vec;

// dp is d rounded up to next power of 2, needed for hadamard transform
// rsign and ssign are the random +1/-1 diagonal matrices for the two SRHTs
typedef struct {
    int   d;
    int   dp;
    int   bits;
    int  *rsign;
    int  *ssign;
    tq_cb cb;
} tq_ctx;

int    tq_init(tq_ctx *ctx, int d, int bits, unsigned int seed);
void   tq_free(tq_ctx *ctx);
tq_vec tq_compress(tq_ctx *ctx, float *v);
void   tq_vec_free(tq_vec *cv);
float  tq_dot(tq_ctx *ctx, float *query, tq_vec *ck);
void   tq_decompress(tq_ctx *ctx, tq_vec *cv, float *out);

#endif
