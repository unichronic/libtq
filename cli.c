#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tq.h"

static float *load_vecs(const char *path, int *n, int *d) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    const char *ext = strrchr(path, '.');
    float *buf = NULL;
    if (ext && strcmp(ext, ".fvecs") == 0) {
        fread(d, 4, 1, f);
        fseek(f, 0, SEEK_END);
        long fsz = ftell(f);
        *n = fsz / (4 + (*d)*4);
        buf = malloc((*n)*(*d)*4);
        rewind(f);
        for (int i = 0; i < *n; i++) {
            int tmp; fread(&tmp, 4, 1, f);
            fread(buf + i*(*d), 4, *d, f);
        }
    } else {
        fread(n, 4, 1, f);
        fread(d, 4, 1, f);
        buf = malloc((*n)*(*d)*4);
        fread(buf, 4, (*n)*(*d), f);
    }
    fclose(f);
    return buf;
}

static void normalize(float *vecs, int n, int d) {
    for (int i = 0; i < n; i++) {
        float *v = vecs+i*d, norm = 0;
        for (int j = 0; j < d; j++) norm += v[j]*v[j];
        norm = sqrtf(norm);
        if (norm > 1e-9f) for (int j = 0; j < d; j++) v[j] /= norm;
    }
}

static void usage(void) {
    puts("tq compress <vecs> <bits> <out.tqb> [-norm]");
    puts("tq dot      <vecs> <bits> <query_idx> [-norm]");
    puts("tq bench    <vecs> <bits> [-norm]");
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }

    if (strcmp(argv[1], "compress") == 0) {
        if (argc < 5) { usage(); return 1; }
        int n, d;
        float *vecs = load_vecs(argv[2], &n, &d);
        if (!vecs) return 1;
        int bits = atoi(argv[3]);
        if (argc > 5 && strcmp(argv[5], "-norm") == 0) normalize(vecs, n, d);
        tq_ctx ctx; tq_init(&ctx, d, bits, 42);
        FILE *out = fopen(argv[4], "wb");
        fwrite(&n, 4, 1, out); fwrite(&d, 4, 1, out); fwrite(&bits, 4, 1, out);
        for (int i = 0; i < n; i++) {
            tq_vec cv = tq_compress(&ctx, vecs+i*d);
            fwrite(cv.idx,    1, d, out);
            fwrite(cv.signs,  1, d, out);
            fwrite(&cv.rnorm, 4, 1, out);
            tq_vec_free(&cv);
        }
        fclose(out);
        long orig = (long)n*d*4, comp = (long)n*(d+d+4);
        printf("compressed %d vecs d=%d bits=%d\n", n, d, bits);
        printf("orig=%ld  comp=%ld  ratio=%.1fx\n", orig, comp, (double)orig/comp);
        tq_free(&ctx); free(vecs);

    } else if (strcmp(argv[1], "dot") == 0) {
        if (argc < 5) { usage(); return 1; }
        int n, d;
        float *vecs = load_vecs(argv[2], &n, &d);
        if (!vecs) return 1;
        int bits = atoi(argv[3]), qi = atoi(argv[4]);
        if (argc > 5 && strcmp(argv[5], "-norm") == 0) normalize(vecs, n, d);
        tq_ctx ctx; tq_init(&ctx, d, bits, 42);
        float *q = vecs+qi*d;
        for (int i = 0; i < 5 && i < n; i++) {
            float true_dot = 0;
            for (int j = 0; j < d; j++) true_dot += q[j]*vecs[i*d+j];
            tq_vec cv = tq_compress(&ctx, vecs+i*d);
            float est = tq_dot(&ctx, q, &cv);
            tq_vec_free(&cv);
            printf("vec[%d]: true=%.4f  tq=%.4f  err=%.4f\n", i, true_dot, est, fabsf(true_dot-est));
        }
        tq_free(&ctx); free(vecs);

    } else if (strcmp(argv[1], "bench") == 0) {
        if (argc < 4) { usage(); return 1; }
        int n, d;
        float *vecs = load_vecs(argv[2], &n, &d);
        if (!vecs) return 1;
        int bits = atoi(argv[3]);
        if (argc > 4 && strcmp(argv[4], "-norm") == 0) normalize(vecs, n, d);
        tq_ctx ctx; tq_init(&ctx, d, bits, 42);
        float *q = vecs, tot_err = 0, tot_true = 0;
        for (int i = 0; i < n; i++) {
            float true_dot = 0;
            for (int j = 0; j < d; j++) true_dot += q[j]*vecs[i*d+j];
            tq_vec cv = tq_compress(&ctx, vecs+i*d);
            tot_err  += fabsf(true_dot - tq_dot(&ctx, q, &cv));
            tot_true += fabsf(true_dot);
            tq_vec_free(&cv);
        }
        printf("d=%d bits=%d n=%d  rel_err=%.3f\n", d, bits, n, tot_err/tot_true);
        tq_free(&ctx); free(vecs);

    } else { usage(); return 1; }
    return 0;
}
