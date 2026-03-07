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
static void solve_lloyd_max(int d, int bits, float *cents, float *bounds) {
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

static float rng(unsigned int *s) {
    *s = *s*1664525u + 1013904223u;
    return ((int)*s) / 2147483648.0f;
}
static void make_rotation(int d, float *Q, unsigned int seed) {
    unsigned int s = seed;
    for (int i = 0; i < d*d; i++) {
        float u1 = (rng(&s)+1.0f)*0.5f+1e-9f, u2 = (rng(&s)+1.0f)*0.5f;
        Q[i] = sqrtf(-2.0f*logf(u1))*cosf(2*3.14159265f*u2);
    }
    for (int i = 0; i < d; i++) {
        float *qi = Q+i*d;
        for (int j = 0; j < i; j++) {
            float *qj = Q+j*d, dot = 0;
            for (int k = 0; k < d; k++) dot += qi[k]*qj[k];
            for (int k = 0; k < d; k++) qi[k] -= dot*qj[k];
        }
        float norm = 0;
        for (int k = 0; k < d; k++) norm += qi[k]*qi[k];
        norm = sqrtf(norm);
        for (int k = 0; k < d; k++) qi[k] /= norm;
    }
}
static void mat_vec(float *M, float *v, float *out, int d) {
    for (int i = 0; i < d; i++) {
        float s = 0;
        for (int j = 0; j < d; j++) s += M[i*d+j]*v[j];
        out[i] = s;
    }
}
static int nearest_cent(tq_cb *cb, float x) {
    int lo = 0, hi = cb->nlvl-1;
    for (int i = 0; i < cb->nlvl-1; i++)
        if (x < cb->bounds[i]) { hi = i; break; } else lo = i+1;
    int best = lo; float bd = fabsf(x-cb->cents[lo]);
    for (int i = lo+1; i <= hi; i++) {
        float dd = fabsf(x-cb->cents[i]);
        if (dd < bd) { bd = dd; best = i; }
    }
    return best;
}

int tq_init(tq_ctx *ctx, int d, int bits, unsigned int seed) {
    ctx->d = d; ctx->bits = bits;
    int nlvl = 1 << bits;
    ctx->cb.nlvl   = nlvl;
    ctx->cb.cents  = malloc(nlvl*sizeof(float));
    ctx->cb.bounds = malloc((nlvl-1)*sizeof(float));
    ctx->rot = malloc(d*d*sizeof(float));
    solve_lloyd_max(d, bits, ctx->cb.cents, ctx->cb.bounds);
    make_rotation(d, ctx->rot, seed);
    return 0;
}
void tq_free(tq_ctx *ctx) {
    free(ctx->cb.cents); free(ctx->cb.bounds); free(ctx->rot);
}

tq_vec tq_compress(tq_ctx *ctx, float *v) {
    int d = ctx->d;
    tq_vec cv; cv.d = d; cv.bits = ctx->bits;
    cv.idx = malloc(d*sizeof(uint8_t));
    float *rotv = malloc(d*sizeof(float));
    float *recon = malloc(d*sizeof(float));
    mat_vec(ctx->rot, v, rotv, d);
    for (int i = 0; i < d; i++) {
        cv.idx[i] = nearest_cent(&ctx->cb, rotv[i]);
        recon[i]  = ctx->cb.cents[cv.idx[i]];
    }
    float *recon_orig = malloc(d*sizeof(float));
    for (int j = 0; j < d; j++) {
        float s = 0;
        for (int i = 0; i < d; i++) s += ctx->rot[i*d+j]*recon[i];
        recon_orig[j] = s;
    }
    float rn = 0;
    for (int i = 0; i < d; i++) { float r = v[i]-recon_orig[i]; rn += r*r; }
    cv.rnorm = sqrtf(rn);
    free(rotv); free(recon); free(recon_orig);
    return cv;
}
void tq_decompress(tq_ctx *ctx, tq_vec *cv, float *out) {
    int d = ctx->d;
    float *recon = malloc(d*sizeof(float));
    for (int i = 0; i < d; i++) recon[i] = ctx->cb.cents[cv->idx[i]];
    for (int j = 0; j < d; j++) {
        float s = 0;
        for (int i = 0; i < d; i++) s += ctx->rot[i*d+j]*recon[i];
        out[j] = s;
    }
    free(recon);
}
void tq_vec_free(tq_vec *cv) { free(cv->idx); }
