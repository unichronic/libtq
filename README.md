# libtq

C library for compressing high-dimensional float vectors to 2-4 bits per dimension and searching over them without decompressing. Uses a two-stage algorithm — optimal scalar quantization after a random rotation, plus a 1-bit residual correction that keeps dot product estimates unbiased.

Useful anywhere you need to store a lot of vectors cheaply and still run fast approximate similarity search over them — embedding databases, vector search engines, or as a building block for ANN indexes.

## How it works

Stage 1: randomly rotates the input vector then applies a Lloyd-Max scalar quantizer per coordinate. The rotation makes each coordinate follow ~N(0, 1/d), which lets a single precomputed codebook work optimally for any input.

Stage 2: computes the residual (original minus reconstruction), projects it through a random Gaussian matrix, and stores only the sign of each projection (1 bit per dimension). This corrects the dot product bias introduced by stage 1.

The result is an unbiased inner product estimator that works directly on compressed vectors without decompressing them.

## Files

```
tq.h        public API
tq.c        implementation
cli.c       command line tool
test_tq.c   tests
Makefile
```

## Build

```
make
```

## Usage

Build a searchable index:
```
./tq build vecs.bin <bits> index.tqb [-norm]
```

Search the index:
```
./tq search index.tqb query.bin <topk> [-norm]
```

Check dot product accuracy:
```
./tq dot vecs.bin <bits> <query_idx>
```

Benchmark error across all vectors:
```
./tq bench vecs.bin <bits>
```

`bits` is typically 2, 3, or 4. Higher = better accuracy, less compression.

Input format for `vecs.bin`: `[N: int32][D: int32][N*D float32]`

## Run tests

```
./test_tq
```

## API

```c
int    tq_init(tq_ctx *ctx, int d, int bits, unsigned int seed);
void   tq_free(tq_ctx *ctx);
tq_vec tq_compress(tq_ctx *ctx, float *v);
void   tq_vec_free(tq_vec *cv);
float  tq_dot(tq_ctx *ctx, float *query, tq_vec *ck);
void   tq_decompress(tq_ctx *ctx, tq_vec *cv, float *out);

int    tq_db_build(tq_db *db, float *vecs, int n, int d, int bits, unsigned int seed);
void   tq_db_free(tq_db *db);
void   tq_search(tq_db *db, float *query, int topk, int *results);
int    tq_db_save(tq_db *db, const char *path);
int    tq_db_load(tq_db *db, const char *path);
```
