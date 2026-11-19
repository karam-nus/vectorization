// Vector-length-agnostic saxpy for RISC-V V 1.0. Compile-checked with:
//   clang --target=riscv64-linux-gnu -march=rv64gcv -ffreestanding -O2 -c saxpy_rvv.c
#include <riscv_vector.h>
#include <stddef.h>

void saxpy_rvv(size_t n, float a, const float *x, float *y) {
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m8(n);             // hardware says how many this time
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);  // m8: a group of 8 registers
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(y, vl);
        vy = __riscv_vfmacc_vf_f32m8(vy, a, vx, vl);     // vy += a * vx
        __riscv_vse32_v_f32m8(y, vy, vl);
        x += vl; y += vl; n -= vl;                       // the last iteration just has a smaller vl
    }
}
