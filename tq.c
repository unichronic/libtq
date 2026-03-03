#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "tq.h"

float l2dist(float *a, float *b, int d) {
    float s = 0;
    for (int i = 0; i < d; i++) { float diff = a[i]-b[i]; s += diff*diff; }
    return sqrtf(s);
}

float *load_fvecs(const char *path, int *n, int *d) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fread(d, 4, 1, f);
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    *n = fsz / (4 + (*d)*4);
    float *buf = malloc((*n)*(*d)*4);
    rewind(f);
    for (int i = 0; i < *n; i++) {
        int tmp; fread(&tmp, 4, 1, f);
        fread(buf + i*(*d), 4, *d, f);
    }
    fclose(f);
    return buf;
}

static float gpdf(float x, float s2) {
    return expf(-x*x/(2*s2)) / sqrtf(2*3.14159265f*s2);
}

static void quad(float a, float b, float s2, float *num, float *den) {
    int steps = 400;
    float h = (b-a)/steps, sn = 0, sd = 0;
    for (int i = 0; i <= steps; i++) {
        float x = a+i*h, p = gpdf(x, s2);
        float w = (i==0||i==steps) ? 0.5f : 1.0f;
        sn += w*x*p; sd += w*p;
    }
    *num = sn*h; *den = sd*h;
}

void solve_lloyd_max(int d, int bits, float *cents, float *bounds) {
    int nlvl = 1 << bits;
    float s2 = 1.0f/d, sig = sqrtf(s2);
    float lo = -3.5f*sig, hi = 3.5f*sig;
    for (int i = 0; i < nlvl; i++)
        cents[i] = lo + (hi-lo)*(i+0.5f)/nlvl;
    for (int iter = 0; iter < 300; iter++) {
        for (int i = 0; i < nlvl-1; i++) bounds[i] = (cents[i]+cents[i+1])*0.5f;
        float nc[256], edges[257];
        edges[0] = lo*3;
        for (int i = 0; i < nlvl-1; i++) edges[i+1] = bounds[i];
        edges[nlvl] = hi*3;
        float ms = 0;
        for (int i = 0; i < nlvl; i++) {
            float num, den;
            quad(edges[i], edges[i+1], s2, &num, &den);
            nc[i] = (den > 1e-15f) ? num/den : cents[i];
            float sh = fabsf(nc[i]-cents[i]);
            if (sh > ms) ms = sh;
        }
        memcpy(cents, nc, nlvl*sizeof(float));
        if (ms < 1e-10f) break;
    }
    for (int i = 0; i < nlvl-1; i++) bounds[i] = (cents[i]+cents[i+1])*0.5f;
}

int tq_init(tq_ctx *ctx, int d, int bits) {
    ctx->d = d; ctx->bits = bits;
    int nlvl = 1 << bits;
    ctx->cb.nlvl   = nlvl;
    ctx->cb.cents  = malloc(nlvl*sizeof(float));
    ctx->cb.bounds = malloc((nlvl-1)*sizeof(float));
    solve_lloyd_max(d, bits, ctx->cb.cents, ctx->cb.bounds);
    return 0;
}

void tq_free(tq_ctx *ctx) {
    free(ctx->cb.cents);
    free(ctx->cb.bounds);
}
