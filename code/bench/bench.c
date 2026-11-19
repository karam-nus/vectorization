/* ============================================================
   bench.c — the kernels behind every measured number in the book.

   The same source is compiled several ways by tools/measurements/run_bench.py
   (scalar, SSE2, AVX2, AVX-512, AVX-512 + -ffast-math). Hand-written AVX-512
   intrinsics live in the same file behind #ifdef __AVX512F__.

   Usage:  ./bench <kernel> <n> [stride]
   Prints one JSON object: {"variant": ..., "kernel": ..., "n": ..., "sec": ..., "checksum": ...}
   ============================================================ */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifndef VARIANT
#define VARIANT "unknown"
#endif
#define NOINLINE __attribute__((noinline))

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* ---------------------------------------------------------------- map kernels */

/* y = a*x + y. restrict promises no aliasing, so no runtime check is needed. */
NOINLINE void saxpy(long n, float a, const float *restrict x, float *restrict y) {
  for (long i = 0; i < n; i++) y[i] = a * x[i] + y[i];
}

/* Leaky ReLU: a branch in the source, a blend in the vector code. */
NOINLINE void leaky(long n, const float *restrict x, float *restrict y) {
  for (long i = 0; i < n; i++) {
    float v = x[i];
    y[i] = v > 0.0f ? v : 0.01f * v;
  }
}

/* Strided read, contiguous write: y[i] = 2 * x[i*s]. */
NOINLINE void strided(long n, long s, const float *restrict x, float *restrict y) {
  for (long i = 0; i < n; i++) y[i] = 2.0f * x[i * s];
}

/* ---------------------------------------------------------------- reductions */

/* In-order float sum. Vectorizing it reorders the additions, which IEEE-754
   forbids unless the compiler is told reassociation is allowed (-ffast-math). */
NOINLINE float dot(long n, const float *restrict x, const float *restrict y) {
  float s = 0.0f;
  for (long i = 0; i < n; i++) s += x[i] * y[i];
  return s;
}

/* Loop-carried dependence of distance 1: y[i] depends on y[i-1]. */
NOINLINE void prefix(long n, const float *restrict x, float *restrict y) {
  float run = 0.0f;
  for (long i = 0; i < n; i++) {
    run += x[i];
    y[i] = run;
  }
}

/* ---------------------------------------------------------------- data layout */

typedef struct { float x, y, z, vx, vy, vz, m, q; } particle; /* 32 bytes */

NOINLINE void step_aos(long n, particle *restrict p, float dt) {
  for (long i = 0; i < n; i++) {
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
  }
}

NOINLINE void step_soa(long n, float *restrict x, float *restrict y, float *restrict z,
                       const float *restrict vx, const float *restrict vy,
                       const float *restrict vz, float dt) {
  for (long i = 0; i < n; i++) {
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
  }
}

/* ---------------------------------------------------------------- matmul loop order */

/* ijk: the innermost loop walks B down a column (stride N) and is a reduction. */
NOINLINE void mm_ijk(long N, const float *restrict A, const float *restrict B, float *restrict C) {
  for (long i = 0; i < N; i++)
    for (long j = 0; j < N; j++) {
      float acc = 0.0f;
      for (long k = 0; k < N; k++) acc += A[i * N + k] * B[k * N + j];
      C[i * N + j] = acc;
    }
}

/* ikj: same arithmetic, innermost loop is a contiguous map over a row of B and C. */
NOINLINE void mm_ikj(long N, const float *restrict A, const float *restrict B, float *restrict C) {
  memset(C, 0, sizeof(float) * (size_t)(N * N));
  for (long i = 0; i < N; i++)
    for (long k = 0; k < N; k++) {
      float a = A[i * N + k];
      for (long j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
    }
}

/* ---------------------------------------------------------------- pairwise distances (Chapter 20) */

/* D[i][j] = sum_k (X[i][k] - Y[j][k])^2, the textbook triple loop. */
NOINLINE void pdist_naive(long n, long m, long d, const float *restrict X,
                          const float *restrict Y, float *restrict D) {
  for (long i = 0; i < n; i++)
    for (long j = 0; j < m; j++) {
      float acc = 0.0f;
      for (long k = 0; k < d; k++) {
        float t = X[i * d + k] - Y[j * d + k];
        acc += t * t;
      }
      D[i * m + j] = acc;
    }
}

/* Same maths with Y stored transposed (YT is d x m): the inner loop runs over j,
   is a map, and vectorizes without reassociation. */
NOINLINE void pdist_outer(long n, long m, long d, const float *restrict X,
                          const float *restrict YT, float *restrict D) {
  for (long i = 0; i < n; i++) {
    float *row = D + i * m;
    for (long j = 0; j < m; j++) row[j] = 0.0f;
    for (long k = 0; k < d; k++) {
      float xv = X[i * d + k];
      const float *yk = YT + k * m;
      for (long j = 0; j < m; j++) {
        float t = xv - yk[j];
        row[j] += t * t;
      }
    }
  }
}

/* ---------------------------------------------------------------- peak FMA throughput (Chapter 9 roofline)
   12 independent accumulator chains, no memory traffic: measures the compute roof for one
   vector width. GCC vector extensions fix the width regardless of -march defaults. */
typedef float v4f __attribute__((vector_size(16)));
typedef float v8f __attribute__((vector_size(32)));
typedef float v16f __attribute__((vector_size(64)));
#define PEAK_KERNEL(NAME, T, ATTR)                                               \
  NOINLINE ATTR float NAME(long iters) {                                         \
    T acc[12], m, a;                                                             \
    for (int k = 0; k < 12; k++) acc[k] = (T){0} + (float)k * 1e-3f;             \
    m = (T){0} + 0.999999f; a = (T){0} + 1e-7f;                                  \
    for (long i = 0; i < iters; i++) {                                           \
      _Pragma("GCC unroll 12")                                                   \
      for (int k = 0; k < 12; k++) acc[k] = acc[k] * m + a;                      \
    }                                                                            \
    T t = acc[0];                                                                \
    for (int k = 1; k < 12; k++) t += acc[k];                                    \
    float r = 0; for (unsigned l = 0; l < sizeof(T) / sizeof(float); l++) r += ((float *)&t)[l]; \
    return r;                                                                    \
  }
typedef float v1f;
/* the scalar roof must stay scalar: switch off the SLP vectorizer for this one function */
PEAK_KERNEL(peak_w1, v1f, __attribute__((optimize("no-tree-slp-vectorize", "no-tree-vectorize"))))
#if defined(__AVX512F__)
PEAK_KERNEL(peak_w4, v4f, )
PEAK_KERNEL(peak_w8, v8f, )
PEAK_KERNEL(peak_w16, v16f, )
#endif

/* ---------------------------------------------------------------- AVX-512 intrinsics */
#ifdef __AVX512F__

/* One accumulator: every FMA waits for the previous one (latency-bound). */
NOINLINE float dot_avx512_1acc(long n, const float *x, const float *y) {
  __m512 acc = _mm512_setzero_ps();
  long i = 0;
  for (; i + 16 <= n; i += 16)
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i), acc);
  __mmask16 tail = (__mmask16)((1u << (n - i)) - 1u);
  acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, x + i), _mm512_maskz_loadu_ps(tail, y + i), acc);
  return _mm512_reduce_add_ps(acc);
}

/* K independent accumulators hide the FMA latency. */
#define DOT_KACC(K)                                                                     \
  NOINLINE float dot_avx512_##K##acc(long n, const float *x, const float *y) {          \
    __m512 acc[K];                                                                      \
    for (int k = 0; k < K; k++) acc[k] = _mm512_setzero_ps();                           \
    long i = 0;                                                                         \
    for (; i + 16 * K <= n; i += 16 * K)                                                \
      for (int k = 0; k < K; k++)                                                       \
        acc[k] = _mm512_fmadd_ps(_mm512_loadu_ps(x + i + 16 * k),                       \
                                 _mm512_loadu_ps(y + i + 16 * k), acc[k]);              \
    for (; i + 16 <= n; i += 16)                                                        \
      acc[0] = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i), acc[0]); \
    __mmask16 tail = (__mmask16)((1u << (n - i)) - 1u);                                 \
    acc[0] = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, x + i),                        \
                             _mm512_maskz_loadu_ps(tail, y + i), acc[0]);               \
    for (int k = 1; k < K; k++) acc[0] = _mm512_add_ps(acc[0], acc[k]);                 \
    return _mm512_reduce_add_ps(acc[0]);                                                \
  }
DOT_KACC(2)
DOT_KACC(4)
DOT_KACC(8)

/* Leaky ReLU with an explicit compare-to-mask and blend; masked tail, no scalar epilogue. */
NOINLINE void leaky_avx512(long n, const float *x, float *y) {
  const __m512 zero = _mm512_setzero_ps(), slope = _mm512_set1_ps(0.01f);
  long i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(x + i);
    __mmask16 pos = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);
    _mm512_storeu_ps(y + i, _mm512_mask_blend_ps(pos, _mm512_mul_ps(v, slope), v));
  }
  __mmask16 tail = (__mmask16)((1u << (n - i)) - 1u);
  __m512 v = _mm512_maskz_loadu_ps(tail, x + i);
  __mmask16 pos = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);
  _mm512_mask_storeu_ps(y + i, tail, _mm512_mask_blend_ps(pos, _mm512_mul_ps(v, slope), v));
}

/* Strided read with a hardware gather (vpgatherdps). */
NOINLINE void strided_gather(long n, long s, const float *x, float *y) {
  const __m512i idx = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                         _mm512_set1_epi32((int)s));
  const __m512 two = _mm512_set1_ps(2.0f);
  long i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(y + i, _mm512_mul_ps(two, _mm512_i32gather_ps(idx, x + i * s, 4)));
  for (; i < n; i++) y[i] = 2.0f * x[i * s];
}
#endif

/* ---------------------------------------------------------------- harness */

static float *alloc_f(size_t count) {
  void *p = NULL;
  if (posix_memalign(&p, 64, count * sizeof(float) + 64)) { perror("alloc"); exit(1); }
  return (float *)p;
}
static void fill(float *a, size_t count, unsigned seed) {
  unsigned s = seed * 2654435761u + 1u;
  for (size_t i = 0; i < count; i++) {
    s = s * 1664525u + 1013904223u;
    a[i] = ((float)(s >> 8) / 16777216.0f) * 2.0f - 1.0f; /* uniform [-1, 1) */
  }
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s kernel n [stride]\n", argv[0]); return 2; }
  const char *k = argv[1];
  long n = atol(argv[2]);
  long s = argc > 3 ? atol(argv[3]) : 1;
  double target = argc > 4 ? atof(argv[4]) : 0.15; /* seconds per trial */

  /* pdist uses n points x n points x 64 dims; mm uses an n x n matrix. */
  long d = 64;
  size_t cap = (size_t)n * (size_t)(s > 1 ? s : 1);
  if (!strcmp(k, "mm_ijk") || !strcmp(k, "mm_ikj")) cap = (size_t)n * (size_t)n;
  if (!strncmp(k, "pdist", 5)) cap = (size_t)n * (size_t)(n > d ? n : d);
  float *x = alloc_f(cap), *y = alloc_f(cap), *z = alloc_f(cap);
  fill(x, cap, 1); fill(y, cap, 2); fill(z, cap, 3);
  particle *P = NULL;
  float *sx = NULL, *sy = NULL, *sz = NULL, *svx = NULL, *svy = NULL, *svz = NULL;
  if (!strncmp(k, "step_", 5)) {
    P = (particle *)alloc_f((size_t)n * 8);
    fill((float *)P, (size_t)n * 8, 4);
    sx = alloc_f(n); sy = alloc_f(n); sz = alloc_f(n); svx = alloc_f(n); svy = alloc_f(n); svz = alloc_f(n);
    fill(sx, n, 5); fill(sy, n, 6); fill(sz, n, 7); fill(svx, n, 8); fill(svy, n, 9); fill(svz, n, 10);
  }

  volatile float sink = 0.0f;
  float last = 0.0f;
  int known = 1;
#define RUN(expr)                                   \
  do {                                              \
    long reps = 1;                                  \
    double best = 1e30;                             \
    for (;;) { /* calibrate reps */                 \
      double t0 = now();                            \
      for (long r = 0; r < reps; r++) { expr; }     \
      double dt = now() - t0;                       \
      if (dt > target / 4 || reps > (1L << 30)) break; \
      reps *= 2;                                    \
    }                                               \
    for (int trial = 0; trial < 5; trial++) {       \
      double t0 = now();                            \
      for (long r = 0; r < reps; r++) { expr; }     \
      double dt = (now() - t0) / (double)reps;      \
      if (dt < best) best = dt;                     \
    }                                               \
    sec = best;                                     \
  } while (0)

  double sec = 0.0;
  if (!strcmp(k, "saxpy")) RUN(saxpy(n, 1.0001f, x, y));
  else if (!strcmp(k, "leaky")) RUN(leaky(n, x, y));
  else if (!strcmp(k, "strided")) RUN(strided(n, s, x, y));
  else if (!strcmp(k, "dot")) RUN(last = dot(n, x, y); sink += last);
  else if (!strcmp(k, "prefix")) RUN(prefix(n, x, y));
  else if (!strcmp(k, "step_aos")) RUN(step_aos(n, P, 1e-3f));
  else if (!strcmp(k, "step_soa")) RUN(step_soa(n, sx, sy, sz, svx, svy, svz, 1e-3f));
  else if (!strcmp(k, "mm_ijk")) RUN(mm_ijk(n, x, y, z));
  else if (!strcmp(k, "mm_ikj")) RUN(mm_ikj(n, x, y, z));
  else if (!strcmp(k, "pdist_naive")) RUN(pdist_naive(n, n, d, x, y, z));
  else if (!strcmp(k, "pdist_outer")) RUN(pdist_outer(n, n, d, x, y, z));
#ifdef __AVX512F__
  else if (!strcmp(k, "dot_avx512_1acc")) RUN(last = dot_avx512_1acc(n, x, y); sink += last);
  else if (!strcmp(k, "dot_avx512_2acc")) RUN(last = dot_avx512_2acc(n, x, y); sink += last);
  else if (!strcmp(k, "dot_avx512_4acc")) RUN(last = dot_avx512_4acc(n, x, y); sink += last);
  else if (!strcmp(k, "dot_avx512_8acc")) RUN(last = dot_avx512_8acc(n, x, y); sink += last);
  else if (!strcmp(k, "leaky_avx512")) RUN(leaky_avx512(n, x, y));
  else if (!strcmp(k, "strided_gather")) RUN(strided_gather(n, s, x, y));
#endif
  else if (!strcmp(k, "peak_w1")) RUN(sink += peak_w1(n));
#ifdef __AVX512F__
  else if (!strcmp(k, "peak_w4")) RUN(sink += peak_w4(n));
  else if (!strcmp(k, "peak_w8")) RUN(sink += peak_w8(n));
  else if (!strcmp(k, "peak_w16")) RUN(sink += peak_w16(n));
#endif
  else known = 0;
  if (!known) { fprintf(stderr, "unknown kernel %s for variant %s\n", k, VARIANT); return 3; }

  /* A checksum the reader can compare across variants: reductions differ when reordered. */
  double check = 0.0;
  if (!strncmp(k, "dot", 3)) check = (double)last;
  else for (long i = 0; i < (n < 4096 ? n : 4096); i++) check += (double)y[i] + (double)z[i];
  printf("{\"variant\": \"%s\", \"kernel\": \"%s\", \"n\": %ld, \"stride\": %ld, \"sec\": %.6e, \"checksum\": %.9g}\n",
         VARIANT, k, n, s, sec, check);
  return 0;
}
