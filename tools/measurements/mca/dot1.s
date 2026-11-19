# one accumulator: every FMA waits for the previous one
vfmadd231ps (%rdi,%rax), %zmm1, %zmm0
vfmadd231ps 64(%rdi,%rax), %zmm2, %zmm0
vfmadd231ps 128(%rdi,%rax), %zmm3, %zmm0
vfmadd231ps 192(%rdi,%rax), %zmm4, %zmm0
