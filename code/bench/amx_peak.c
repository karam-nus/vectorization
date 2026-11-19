/* amx_peak.c — register-only throughput of Intel AMX bf16 tile multiplies (Chapter 16).

   gcc -O2 -mamx-tile -mamx-bf16 -o amx_peak amx_peak.c && ./amx_peak
   Prints one JSON object. Needs Linux >= 5.16 (arch_prctl permission for tile data).

   One TDPBF16PS: C[16x16 fp32] += A[16x32 bf16] . B[32x16 bf16] = 16*16*32 MACs = 16384 FLOP.
   Four independent accumulators (tmm0-3) from two A tiles x two B tiles hide the latency. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

typedef struct {
  uint8_t palette_id, start_row, reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __attribute__((packed)) tilecfg;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static uint16_t A[2][16 * 32], B[2][16 * 32];  /* bf16 bit patterns */
static float C[4][16 * 16];

int main(void) {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) { perror("arch_prctl"); return 1; }
  tilecfg cfg; memset(&cfg, 0, sizeof cfg);
  cfg.palette_id = 1;
  for (int t = 0; t < 8; t++) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
  _tile_loadconfig(&cfg);
  for (int i = 0; i < 16 * 32; i++) { A[0][i] = A[1][i] = 0x3c00; B[0][i] = B[1][i] = 0x3c00; } /* tiny bf16 */
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  _tile_loadd(4, A[0], 64); _tile_loadd(5, A[1], 64); _tile_loadd(6, B[0], 64); _tile_loadd(7, B[1], 64);
  long iters = 2000000;
  double best = 1e30;
  for (int trial = 0; trial < 5; trial++) {
    double t0 = now();
    for (long i = 0; i < iters; i++) {
      _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(1, 4, 7);
      _tile_dpbf16ps(2, 5, 6); _tile_dpbf16ps(3, 5, 7);
    }
    double dt = now() - t0;
    if (dt < best) best = dt;
  }
  _tile_stored(0, C[0], 64); _tile_stored(1, C[1], 64); _tile_stored(2, C[2], 64); _tile_stored(3, C[3], 64);
  _tile_release();
  double flop = (double)iters * 4 * 16 * 16 * 32 * 2;
  printf("{\"kernel\": \"amx_tdpbf16ps_4acc\", \"iters\": %ld, \"sec\": %.6e, \"gflops\": %.1f, \"checksum\": %.6g}\n",
         iters, best, flop / best / 1e9, (double)C[0][0] + C[3][255]);
  return 0;
}
