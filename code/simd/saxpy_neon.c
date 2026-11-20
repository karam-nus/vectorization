// Fixed-width 128-bit saxpy for Arm NEON (AArch64). Compile-checked with:
//   clang --target=aarch64-linux-gnu -ffreestanding -O2 -c saxpy_neon.c
#include <arm_neon.h>
#include <stdint.h>

void saxpy_neon(int64_t n, float a, const float *x, float *y) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vy = vld1q_f32(y + i);
        vy = vfmaq_n_f32(vy, vld1q_f32(x + i), a);       // vy + vx * a, 4 lanes
        vst1q_f32(y + i, vy);
    }
    for (; i < n; i++) y[i] += a * x[i];                 // scalar tail: NEON has no masked store
}
