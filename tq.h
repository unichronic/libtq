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

// compressed database - flat array of n compressed vectors + the context needed to search
typedef struct {
    tq_ctx   ctx;
    int      n;
    uint8_t *ibuf;   // all indices packed: n * ceil(d*bits/8) bytes
    uint8_t *sbuf;   // all signs packed:   n * ceil(d/8) bytes
    float   *rnorms; // residual norms:     n floats
} tq_db;

int    tq_init(tq_ctx *ctx, int d, int bits, unsigned int seed);
void   tq_free(tq_ctx *ctx);
tq_vec tq_compress(tq_ctx *ctx, float *v);
void   tq_vec_free(tq_vec *cv);
float  tq_dot(tq_ctx *ctx, float *query, tq_vec *ck);
void   tq_decompress(tq_ctx *ctx, tq_vec *cv, float *out);

// build a searchable compressed db from n raw vectors
int  tq_db_build(tq_db *db, float *vecs, int n, int d, int bits, unsigned int seed);
void tq_db_free(tq_db *db);

// top-k search, results filled with indices of nearest vectors (by dot product)
void tq_search(tq_db *db, float *query, int topk, int *results);

// save/load db to disk
int tq_db_save(tq_db *db, const char *path);
int tq_db_load(tq_db *db, const char *path);

#endif
