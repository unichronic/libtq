# libtq

C library for compressing high-dimensional float vectors to 2-4 bits per dimension and searching over them without decompressing. Uses a two-stage algorithm: optimal scalar quantization after a random rotation, plus a 1-bit residual correction that keeps dot product estimates unbiased.

Useful anywhere you need to store a lot of vectors cheaply and still run fast approximate similarity search over them: embedding databases, vector search engines, or as a building block for ANN indexes.

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

Step 1 - build an index from your vectors:
```
./tq build vecs.fvecs 3 index.tqb
```

Step 2 - search the index with a query file:
```
./tq search index.tqb query.fvecs 10
```

That's it. The index file can be reloaded and searched without rebuilding.

`bits` controls the compression/accuracy tradeoff: 2 = most compressed, 4 = most accurate. 3 is a good default.

Add `-norm` if your vectors are not already unit-normalized (e.g. raw SIFT descriptors). Embedding vectors from models are usually already normalized.

Supported input formats: `.fvecs`, `.npy`, `.fbin`, raw binary `[N: int32][D: int32][N*D float32]`

### Debug commands

Check how accurate the dot product estimates are:
```
./tq dot vecs.fvecs 3 0
```

Measure average compression error across all vectors:
```
./tq bench vecs.fvecs 3
```

## Benchmarks

Tested on a standard CPU. Two datasets:
- **SIFT1M** (100K subset, d=128): raw SIFT descriptors, normalized. Non-Gaussian distribution, harder case.
- **GloVe-100** (100K subset, d=100): word embedding vectors, already unit-normalized. Better fit for the algorithm.

### Compression

**SIFT1M** (d=128, normalized):

| bits | ratio | build time | dot error (relative) |
|------|-------|------------|----------------------|
| 2    | 9.8x  | 0.37s      | 63%                  |
| 3    | 7.5x  | 0.40s      | 54%                  |
| 4    | 6.1x  | 0.46s      | 47%                  |

**GloVe-100** (d=100, unit-normalized embeddings):

| bits | ratio | build time | dot error (relative) |
|------|-------|------------|----------------------|
| 2    | 9.5x  | 0.51s      | 27%                  |
| 3    | 7.3x  | 0.57s      | 23%                  |
| 4    | 6.0x  | 0.65s      | 21%                  |

Higher error on SIFT is expected: SIFT descriptors are histogram-based with non-Gaussian distributions. Embedding vectors (GloVe, sentence embeddings, etc.) fit the algorithm's assumptions much better.

### Search (brute-force, 10 queries over 100K vectors)

| bits | SIFT time | GloVe time |
|------|-----------|------------|
| 2    | 1.2s      | 1.9s       |
| 3    | 1.7s      | 2.7s       |
| 4    | 2.2s      | 3.5s       |

Search is brute-force O(N) over compressed vectors. No decompression during search. Uses precomputed distance lookup tables and unpacked scan buffers to avoid bit manipulation in the hot loop.

## Run tests

```
./test_tq
```
