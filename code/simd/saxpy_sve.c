// Vector-length-agnostic saxpy for Arm SVE. Compile-checked with:
//   clang --target=aarch64-linux-gnu -march=armv8-a+sve -ffreestanding -O2 -c saxpy_sve.c
#include <arm_sve.h>
#include <stdint.h>

void saxpy_sve(int64_t n, float a, const float *x, float *y) {
    for (int64_t i = 0; i < n; i += svcntw()) {          // svcntw(): fp32 lanes on THIS chip
        svbool_t pg = svwhilelt_b32_s64(i, n);           // predicate: lanes i..n-1 active
        svfloat32_t vx = svld1_f32(pg, x + i);           // masked load, no fault past n
        svfloat32_t vy = svld1_f32(pg, y + i);
        vy = svmla_n_f32_m(pg, vy, vx, a);               // vy += vx * a on active lanes
        svst1_f32(pg, y + i, vy);                        // masked store: the tail is free
    }
}
