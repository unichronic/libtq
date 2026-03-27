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
    } else if (ext && strcmp(ext, ".npy") == 0) {
        char magic[6]; fread(magic, 1, 6, f);
        uint8_t major, minor; fread(&major, 1, 1, f); fread(&minor, 1, 1, f);
        uint16_t hlen; fread(&hlen, 2, 1, f);
        char *hdr = malloc(hlen+1); fread(hdr, 1, hlen, f); hdr[hlen] = 0;
        char *sp = strstr(hdr, "("); int n2, d2;
        sscanf(sp, "(%d, %d)", &n2, &d2);
        *n = n2; *d = d2;
        buf = malloc((*n)*(*d)*4);
        fread(buf, 4, (*n)*(*d), f);
        free(hdr);
    } else if (ext && strcmp(ext, ".fbin") == 0) {
        fread(n, 4, 1, f);
        fread(d, 4, 1, f);
        buf = malloc((*n)*(*d)*4);
        fread(buf, 4, (*n)*(*d), f);
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
        float *v = vecs + i*d, norm = 0;
        for (int j = 0; j < d; j++) norm += v[j]*v[j];
        norm = sqrtf(norm);
        if (norm > 1e-9f) for (int j = 0; j < d; j++) v[j] /= norm;
    }
}

static void usage(void) {
    puts("tq build    <vecs> <bits> <out.tqb> [-norm]");
    puts("tq search   <index.tqb> <query_vec> <topk> [-norm]");
    puts("tq compress <vecs> <bits> <out.tqb> [-norm]");
    puts("tq dot      <vecs> <bits> <query_idx> [-norm]");
    puts("tq bench    <vecs> <bits> [-norm]");
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }

    if (strcmp(argv[1], "build") == 0) {
        if (argc < 5) { usage(); return 1; }
        int n, d;
        float *vecs = load_vecs(argv[2], &n, &d);
        if (!vecs) return 1;
        int bits = atoi(argv[3]);
        if (argc > 5 && strcmp(argv[5], "-norm") == 0) normalize(vecs, n, d);
        tq_db db;
        if (tq_db_build(&db, vecs, n, d, bits, 42) != 0) { fprintf(stderr, "build failed\n"); return 1; }
        if (tq_db_save(&db, argv[4]) != 0) return 1;
        long orig = (long)n*d*4;
        long ibytes = (long)n*((d*bits+7)/8);
        long sbytes = (long)n*((d+7)/8);
        long comp = ibytes + sbytes + n*4;
        printf("built index: %d vecs d=%d bits=%d\n", n, d, bits);
        printf("orig=%ld bytes  comp=%ld bytes  ratio=%.1fx\n", orig, comp, (double)orig/comp);
        tq_db_free(&db); free(vecs);

    } else if (strcmp(argv[1], "search") == 0) {
        if (argc < 5) { usage(); return 1; }
        tq_db db;
        if (tq_db_load(&db, argv[2]) != 0) return 1;
        int n, d;
        float *queries = load_vecs(argv[3], &n, &d);
        if (!queries) return 1;
        int topk = atoi(argv[4]);
        if (argc > 5 && strcmp(argv[5], "-norm") == 0) normalize(queries, n, d);
        if (d != db.ctx.d) { fprintf(stderr, "dim mismatch\n"); return 1; }
        int *results = malloc(topk * sizeof(int));
        for (int q = 0; q < n; q++) {
            tq_search(&db, queries + q*d, topk, results);
            printf("query %d -> [", q);
            for (int i = 0; i < topk; i++) printf("%d%s", results[i], i<topk-1?", ":"");
            printf("]\n");
        }
        free(results); free(queries); tq_db_free(&db);

    } else if (strcmp(argv[1], "compress") == 0) {
        if (argc < 5) { usage(); return 1; }
        int n, d;
        float *vecs = load_vecs(argv[2], &n, &d);
        if (!vecs) return 1;
        int bits = atoi(argv[3]);
        if (argc > 5 && strcmp(argv[5], "-norm") == 0) normalize(vecs, n, d);

        tq_ctx ctx;
        tq_init(&ctx, d, bits, 42);

        FILE *out = fopen(argv[4], "wb");
        fwrite(&n, 4, 1, out);
        fwrite(&d, 4, 1, out);
        fwrite(&bits, 4, 1, out);

        for (int i = 0; i < n; i++) {
            tq_vec cv = tq_compress(&ctx, vecs + i*d);
            int ibytes = (d*bits + 7) / 8;
            int sbytes = (d + 7) / 8;
            fwrite(cv.ibuf,     1, ibytes, out);
            fwrite(cv.sbuf,     1, sbytes, out);
            fwrite(&cv.rnorm16, 2, 1,      out);
            tq_vec_free(&cv);
        }
        fclose(out);

        long orig = (long)n*d*4;
        long ibytes = (d*bits + 7) / 8;
        long sbytes = (d + 7) / 8;
        long comp = (long)n*(ibytes + sbytes + 2);
        printf("compressed %d vecs d=%d bits=%d\n", n, d, bits);
        printf("orig=%ld bytes  comp=%ld bytes  ratio=%.1fx\n", orig, comp, (double)orig/comp);

        tq_free(&ctx);
        free(vecs);

    } else if (strcmp(argv[1], "dot") == 0) {
        if (argc < 5) { usage(); return 1; }
        int n, d;
        float *vecs = load_vecs(argv[2], &n, &d);
        if (!vecs) return 1;
        int bits = atoi(argv[3]);
        int qi   = atoi(argv[4]);
        if (argc > 5 && strcmp(argv[5], "-norm") == 0) normalize(vecs, n, d);
        if (qi >= n) { fprintf(stderr, "idx out of range\n"); return 1; }

        tq_ctx ctx;
        tq_init(&ctx, d, bits, 42);

        float *q = vecs + qi*d;
        printf("query idx=%d  true dots vs tq_dot (first 5 vecs):\n", qi);
        for (int i = 0; i < 5 && i < n; i++) {
            float *v = vecs + i*d;
            float true_dot = 0;
            for (int j = 0; j < d; j++) true_dot += q[j]*v[j];
            tq_vec cv = tq_compress(&ctx, v);
            float est = tq_dot(&ctx, q, &cv);
            tq_vec_free(&cv);
            printf("  vec[%d]: true=%.4f  tq=%.4f  err=%.4f\n", i, true_dot, est, fabsf(true_dot-est));
        }

        tq_free(&ctx);
        free(vecs);

    } else if (strcmp(argv[1], "bench") == 0) {
        if (argc < 4) { usage(); return 1; }
        int n, d;
        float *vecs = load_vecs(argv[2], &n, &d);
        if (!vecs) return 1;
        int bits = atoi(argv[3]);
        if (argc > 4 && strcmp(argv[4], "-norm") == 0) normalize(vecs, n, d);

        tq_ctx ctx;
        tq_init(&ctx, d, bits, 42);

        tq_vec *cvs = malloc(n * sizeof(tq_vec));
        for (int i = 0; i < n; i++)
            cvs[i] = tq_compress(&ctx, vecs + i*d);

        float *q = vecs;
        float tot_err = 0, tot_true = 0;
        for (int i = 0; i < n; i++) {
            float true_dot = 0;
            for (int j = 0; j < d; j++) true_dot += q[j]*vecs[i*d+j];
            float est = tq_dot(&ctx, q, &cvs[i]);
            tot_err  += fabsf(true_dot - est);
            tot_true += fabsf(true_dot);
        }
        printf("d=%d bits=%d n=%d  mean_abs_err=%.5f  mean_abs_true=%.5f  rel=%.3f\n",
               d, bits, n, tot_err/n, tot_true/n, tot_err/tot_true);

        for (int i = 0; i < n; i++) tq_vec_free(&cvs[i]);
        free(cvs);
        tq_free(&ctx);
        free(vecs);

    } else {
        usage();
        return 1;
    }
    return 0;
}
