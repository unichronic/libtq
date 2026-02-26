#ifndef TQ_H
#define TQ_H

#include <stdint.h>

typedef struct {
    int    d;
    float *cents;
    int    nlvl;
} tq_cb;

typedef struct {
    int   d;
    int   bits;
    tq_cb cb;
} tq_ctx;

float l2dist(float *a, float *b, int d);
float *load_fvecs(const char *path, int *n, int *d);

#endif
