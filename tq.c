#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "tq.h"

// fp16 conversion - doing manually because dont want to pull in any lib
// formula: sign(1) | exponent(5) | mantissa(10), bias is 15 not 127
static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint16_t s = (x >> 16) & 0x8000;
    int      e = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0)  return s;
    if (e >= 31) return s | 0x7c00;
    return s | (e << 10) | (m >> 13);
}
static float f16_to_f32(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16;
    int      e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ff;
    if (e == 0)  { float f; uint32_t v = s | (m << 13); memcpy(&f, &v, 4); return f; }
    if (e == 31) { float f; uint32_t v = s | 0x7f800000 | (m << 13); memcpy(&f, &v, 4); return f; }
    uint32_t v = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &v, 4); return f;
}

// pack val into buf at position idx, using `bits` bits per value
// storing LSB first, bit position = idx*bits + b
static void pack_bits(uint8_t *buf, int idx, int val, int bits) {
    int bit = idx * bits;
    for (int b = 0; b < bits; b++) {
        int pos = bit + b;
        if ((val >> b) & 1) buf[pos/8] |=  (1 << (pos%8));
        else                buf[pos/8] &= ~(1 << (pos%8));
    }
}
static int unpack_bits(uint8_t *buf, int idx, int bits) {
    int bit = idx * bits, val = 0;
    for (int b = 0; b < bits; b++) {
        int pos = bit + b;
        if ((buf[pos/8] >> (pos%8)) & 1) val |= (1 << b);
    }
    return val;
}

// gaussian pdf with variance s2 = 1/d
// after random rotation each coord of unit vector ~ N(0, 1/d), this is why
// the same codebook works for any input - rotation makes distribution predictable
static float gpdf(float x, float s2) {
    return expf(-x*x/(2*s2)) / sqrtf(2*3.14159265f*s2);
}

// trapezoid integration, 400 steps is enough for convergence
static void quad(float a, float b, float s2, float *num, float *den) {
    int steps = 400;
    float h = (b-a)/steps, sn = 0, sd = 0;
    for (int i = 0; i <= steps; i++) {
        float x = a + i*h, p = gpdf(x, s2);
        float w = (i==0||i==steps) ? 0.5f : 1.0f;
        sn += w*x*p; sd += w*p;
    }
    *num = sn*h; *den = sd*h;
}

// lloyd-max: iteratively find optimal centroids for N(0,1/d) distribution
// centroid update = E[X | X in bucket] = integral(x*pdf) / integral(pdf)
// boundaries are just midpoints between adjacent centroids
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

// fast walsh hadamard transform, O(n log n), n must be power of 2
// butterfly pattern same as FFT but only additions/subtractions, no complex numbers
static void fwht(float *x, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len<<1) {
            for (int j = 0; j < len; j++) {
                float a = x[i+j], b = x[i+j+len];
                x[i+j] = a+b; x[i+j+len] = a-b;
            }
        }
    }
    float sc = 1.0f/sqrtf(n);
    for (int i = 0; i < n; i++) x[i] *= sc;
}

// SRHT = randomized hadamard transform
// multiply by diagonal sign matrix first, then hadamard
// this is the rotation - much faster than full matrix multiply O(n log n) vs O(n^2)
static void srht(float *v, int *signs, float *out, int dp) {
    float *tmp = malloc(dp*sizeof(float));
    for (int i = 0; i < dp; i++) tmp[i] = v[i] * signs[i];
    fwht(tmp, dp);
    memcpy(out, tmp, dp*sizeof(float));
    free(tmp);
}

// transpose of srht - hadamard first then sign flip
// hadamard is its own inverse (up to scaling) so H^T = H
static void srht_T(float *v, int *signs, float *out, int dp) {
    float *tmp = malloc(dp*sizeof(float));
    memcpy(tmp, v, dp*sizeof(float));
    fwht(tmp, dp);
    for (int i = 0; i < dp; i++) out[i] = tmp[i] * signs[i];
    free(tmp);
}

static unsigned int lcg(unsigned int *s) {
    *s = *s*1664525u + 1013904223u;
    return *s;
}

static int next_pow2(int n) {
    int p = 1; while (p < n) p <<= 1; return p;
}

static int nearest_cent(tq_cb *cb, float x) {
    int lo = 0, hi = cb->nlvl-1;
    for (int i = 0; i < cb->nlvl-1; i++)
        if (x < cb->bounds[i]) { hi = i; break; } else lo = i+1;
    int best = lo;
    float bd = fabsf(x-cb->cents[lo]);
    for (int i = lo+1; i <= hi; i++) {
        float dd = fabsf(x-cb->cents[i]);
        if (dd < bd) { bd = dd; best = i; }
    }
    return best;
}

int tq_init(tq_ctx *ctx, int d, int bits, unsigned int seed) {
    ctx->d    = d;
    ctx->dp   = next_pow2(d);
    ctx->bits = bits;
    int nlvl = 1 << bits;
    ctx->cb.nlvl   = nlvl;
    ctx->cb.cents  = malloc(nlvl*sizeof(float));
    ctx->cb.bounds = malloc((nlvl-1)*sizeof(float));
    ctx->rsign = malloc(ctx->dp*sizeof(int));
    ctx->ssign = malloc(ctx->dp*sizeof(int));
    if (!ctx->cb.cents || !ctx->cb.bounds || !ctx->rsign || !ctx->ssign)
        return -1;
    solve_lloyd_max(d, bits, ctx->cb.cents, ctx->cb.bounds);
    unsigned int s = seed;
    for (int i = 0; i < ctx->dp; i++) ctx->rsign[i] = (lcg(&s) & 1) ? 1 : -1;
    s = seed + 99999;
    // different seed for qjl matrix so its independent from rotation
    for (int i = 0; i < ctx->dp; i++) ctx->ssign[i] = (lcg(&s) & 1) ? 1 : -1;
    return 0;
}

void tq_free(tq_ctx *ctx) {
    free(ctx->cb.cents); free(ctx->cb.bounds);
    free(ctx->rsign);    free(ctx->ssign);
}

tq_vec tq_compress(tq_ctx *ctx, float *v) {
    int d = ctx->d, dp = ctx->dp, bits = ctx->bits;
    int ibytes = (d*bits + 7) / 8;
    int sbytes = (d + 7) / 8;
    tq_vec cv;
    cv.d    = d;
    cv.bits = bits;
    cv.ibuf = calloc(ibytes, 1);
    cv.sbuf = calloc(sbytes, 1);
    // pad to dp with zeros for hadamard
    float *vp = calloc(dp, sizeof(float));
    memcpy(vp, v, d*sizeof(float));
    float *rotv = malloc(dp*sizeof(float));
    srht(vp, ctx->rsign, rotv, dp);
    float *recon_rot = malloc(dp*sizeof(float));
    for (int i = 0; i < d; i++) {
        int ci = nearest_cent(&ctx->cb, rotv[i]);
        pack_bits(cv.ibuf, i, ci, bits);
        recon_rot[i] = ctx->cb.cents[ci];
    }
    for (int i = d; i < dp; i++) recon_rot[i] = 0;
    float *recon_orig = malloc(dp*sizeof(float));
    srht_T(recon_rot, ctx->rsign, recon_orig, dp);
    float *resid = malloc(dp*sizeof(float));
    float rnorm2 = 0;
    for (int i = 0; i < d; i++) {
        resid[i] = v[i] - recon_orig[i];
        rnorm2 += resid[i]*resid[i];
    }
    for (int i = d; i < dp; i++) resid[i] = 0;
    cv.rnorm16 = f32_to_f16(sqrtf(rnorm2));
    // project residual thru S and store only sign - this is the qjl part
    // 1 bit per dim is enough to debias the dot product estimate
    float *proj = malloc(dp*sizeof(float));
    srht(resid, ctx->ssign, proj, dp);
    for (int i = 0; i < d; i++) {
        if (proj[i] >= 0) cv.sbuf[i/8] |=  (1 << (i%8));
        else              cv.sbuf[i/8] &= ~(1 << (i%8));
    }
    free(vp); free(rotv); free(recon_rot); free(recon_orig); free(resid); free(proj);
    return cv;
}

void tq_vec_free(tq_vec *cv) {
    free(cv->ibuf); free(cv->sbuf);
}

// unbiased dot product estimator from turboquant paper
// t1 = <rotated_query, codebook_centroids>  (mse part)
// t2 = ||residual|| * sqrt(pi/2) / d * <S*query, signs>  (qjl correction)
// the sqrt(pi/2)/d scaling comes from E[|N(0,1)|] = sqrt(2/pi), need to invert that
float tq_dot(tq_ctx *ctx, float *query, tq_vec *ck) {
    int d = ctx->d, dp = ctx->dp;
    float *qp = calloc(dp, sizeof(float));
    memcpy(qp, query, d*sizeof(float));
    float *rotq = malloc(dp*sizeof(float));
    srht(qp, ctx->rsign, rotq, dp);
    float t1 = 0;
    for (int i = 0; i < d; i++)
        t1 += rotq[i] * ctx->cb.cents[unpack_bits(ck->ibuf, i, ck->bits)];
    float *sq = malloc(dp*sizeof(float));
    srht(qp, ctx->ssign, sq, dp);
    float qjl_ip = 0;
    for (int i = 0; i < d; i++) {
        int sign = ((ck->sbuf[i/8] >> (i%8)) & 1) ? 1 : -1;
        qjl_ip += sq[i] * sign;
    }
    float rnorm = f16_to_f32(ck->rnorm16);
    float scale = sqrtf(3.14159265f/2.0f) / d;
    float t2 = rnorm * scale * qjl_ip;
    free(qp); free(rotq); free(sq);
    return t1 + t2;
}

void tq_decompress(tq_ctx *ctx, tq_vec *cv, float *out) {
    int d = ctx->d, dp = ctx->dp;
    float *recon_rot = calloc(dp, sizeof(float));
    for (int i = 0; i < d; i++)
        recon_rot[i] = ctx->cb.cents[unpack_bits(cv->ibuf, i, cv->bits)];
    float *tmp = malloc(dp*sizeof(float));
    srht_T(recon_rot, ctx->rsign, tmp, dp);
    memcpy(out, tmp, d*sizeof(float));
    free(recon_rot); free(tmp);
}

int tq_db_build(tq_db *db, float *vecs, int n, int d, int bits, unsigned int seed) {
    if (tq_init(&db->ctx, d, bits, seed) != 0) return -1;
    db->n = n;
    int ibytes = (d*bits + 7) / 8;
    int sbytes = (d + 7) / 8;
    db->ibuf   = malloc((long)n * ibytes);
    db->sbuf   = malloc((long)n * sbytes);
    db->rnorms = malloc(n * sizeof(float));
    if (!db->ibuf || !db->sbuf || !db->rnorms) return -1;
    for (int i = 0; i < n; i++) {
        tq_vec cv = tq_compress(&db->ctx, vecs + i*d);
        memcpy(db->ibuf + (long)i*ibytes, cv.ibuf, ibytes);
        memcpy(db->sbuf + (long)i*sbytes, cv.sbuf, sbytes);
        db->rnorms[i] = f16_to_f32(cv.rnorm16);
        tq_vec_free(&cv);
    }
    return 0;
}

void tq_db_free(tq_db *db) {
    tq_free(&db->ctx);
    free(db->ibuf); free(db->sbuf); free(db->rnorms);
}

// dot product directly against db row without making a tq_vec
static float db_dot(tq_db *db, float *rotq, float *sq, int i) {
    int d = db->ctx.d, bits = db->ctx.bits;
    int ibytes = (d*bits + 7) / 8;
    int sbytes = (d + 7) / 8;
    uint8_t *ib = db->ibuf + (long)i*ibytes;
    uint8_t *sb = db->sbuf + (long)i*sbytes;
    float t1 = 0;
    for (int j = 0; j < d; j++)
        t1 += rotq[j] * db->ctx.cb.cents[unpack_bits(ib, j, bits)];
    float qjl = 0;
    for (int j = 0; j < d; j++) {
        int sign = ((sb[j/8] >> (j%8)) & 1) ? 1 : -1;
        qjl += sq[j] * sign;
    }
    float scale = sqrtf(3.14159265f/2.0f) / d;
    return t1 + db->rnorms[i] * scale * qjl;
}

void tq_search(tq_db *db, float *query, int topk, int *results) {
    int d = db->ctx.d, dp = db->ctx.dp;
    int bits = db->ctx.bits, nlvl = db->ctx.cb.nlvl;
    int ibytes = (d*bits+7)/8;
    int sbytes = (d+7)/8;

    float *qp = calloc(dp, sizeof(float));
    memcpy(qp, query, d*sizeof(float));
    float *rotq = malloc(dp*sizeof(float));
    float *sq   = malloc(dp*sizeof(float));
    srht(qp, db->ctx.rsign, rotq, dp);
    srht(qp, db->ctx.ssign, sq,   dp);

    // precompute lut[j*nlvl + c] = rotq[j] * centroid[c]
    // inner loop becomes table lookup instead of multiply
    float *lut = malloc(d * nlvl * sizeof(float));
    for (int j = 0; j < d; j++)
        for (int c = 0; c < nlvl; c++)
            lut[j*nlvl + c] = rotq[j] * db->ctx.cb.cents[c];

    // unpack all codes to flat uint8 once - avoids bit manipulation in hot loop
    // trades memory (n*d bytes) for speed, same trick faiss uses for PQ scan
    uint8_t *codes_flat = malloc((long)db->n * d);
    for (int i = 0; i < db->n; i++)
        for (int j = 0; j < d; j++)
            codes_flat[(long)i*d + j] = unpack_bits(db->ibuf + (long)i*ibytes, j, bits);

    // precompute sq as int8 signs for qjl - avoids bit extract in hot loop
    // sign_flat[i*d+j] = +1 or -1
    int8_t *sign_flat = malloc((long)db->n * d);
    for (int i = 0; i < db->n; i++)
        for (int j = 0; j < d; j++)
            sign_flat[(long)i*d + j] = ((db->sbuf[(long)i*sbytes + j/8] >> (j%8)) & 1) ? 1 : -1;

    float scale = sqrtf(3.14159265f/2.0f) / d;

    float *best_scores = malloc(topk * sizeof(float));
    for (int i = 0; i < topk; i++) { results[i] = -1; best_scores[i] = -1e30f; }
    float worst_score = -1e30f;

    for (int i = 0; i < db->n; i++) {
        uint8_t *codes = codes_flat + (long)i*d;
        int8_t  *signs = sign_flat  + (long)i*d;

        float t1 = 0;
        for (int j = 0; j < d; j++)
            t1 += lut[j*nlvl + codes[j]];

        float qjl = 0;
        for (int j = 0; j < d; j++)
            qjl += sq[j] * signs[j];

        float score = t1 + db->rnorms[i] * scale * qjl;

        if (score > worst_score) {
            int worst = 0;
            for (int j = 1; j < topk; j++)
                if (best_scores[j] < best_scores[worst]) worst = j;
            best_scores[worst] = score;
            results[worst] = i;
            worst_score = best_scores[0];
            for (int j = 1; j < topk; j++)
                if (best_scores[j] < worst_score) worst_score = best_scores[j];
        }
    }

    free(qp); free(rotq); free(sq); free(lut);
    free(codes_flat); free(sign_flat); free(best_scores);
}

// file format: [d,dp,bits,n,nlvl] then codebook cents+bounds, rsign, ssign, ibuf, sbuf, rnorms
int tq_db_save(tq_db *db, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    int d = db->ctx.d, dp = db->ctx.dp, bits = db->ctx.bits;
    int n = db->n, nlvl = db->ctx.cb.nlvl;
    fwrite(&d,    4, 1, f); fwrite(&dp,   4, 1, f);
    fwrite(&bits, 4, 1, f); fwrite(&n,    4, 1, f);
    fwrite(&nlvl, 4, 1, f);
    fwrite(db->ctx.cb.cents,  4, nlvl,   f);
    fwrite(db->ctx.cb.bounds, 4, nlvl-1, f);
    fwrite(db->ctx.rsign, sizeof(int), dp, f);
    fwrite(db->ctx.ssign, sizeof(int), dp, f);
    long ibytes = (long)n * ((d*bits+7)/8);
    long sbytes = (long)n * ((d+7)/8);
    fwrite(db->ibuf,   1, ibytes, f);
    fwrite(db->sbuf,   1, sbytes, f);
    fwrite(db->rnorms, 4, n,      f);
    fclose(f);
    return 0;
}

int tq_db_load(tq_db *db, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    int d, dp, bits, n, nlvl;
    fread(&d, 4, 1, f); fread(&dp,   4, 1, f);
    fread(&bits, 4, 1, f); fread(&n, 4, 1, f);
    fread(&nlvl, 4, 1, f);
    db->ctx.d = d; db->ctx.dp = dp; db->ctx.bits = bits;
    db->ctx.cb.nlvl   = nlvl;
    db->ctx.cb.cents  = malloc(nlvl*4);
    db->ctx.cb.bounds = malloc((nlvl-1)*4);
    db->ctx.rsign = malloc(dp*sizeof(int));
    db->ctx.ssign = malloc(dp*sizeof(int));
    fread(db->ctx.cb.cents,  4, nlvl,   f);
    fread(db->ctx.cb.bounds, 4, nlvl-1, f);
    fread(db->ctx.rsign, sizeof(int), dp, f);
    fread(db->ctx.ssign, sizeof(int), dp, f);
    db->n = n;
    long ibytes = (long)n * ((d*bits+7)/8);
    long sbytes = (long)n * ((d+7)/8);
    db->ibuf   = malloc(ibytes);
    db->sbuf   = malloc(sbytes);
    db->rnorms = malloc(n*4);
    fread(db->ibuf,   1, ibytes, f);
    fread(db->sbuf,   1, sbytes, f);
    fread(db->rnorms, 4, n,      f);
    fclose(f);
    return 0;
}
