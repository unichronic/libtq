#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "tq.h"

float l2dist(float *a, float *b, int d) {
    float s = 0;
    for (int i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        s += diff*diff;
    }
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
