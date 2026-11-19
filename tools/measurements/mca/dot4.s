# four accumulators: independent chains
vfmadd231ps (%rdi,%rax), %zmm1, %zmm0
vfmadd231ps 64(%rdi,%rax), %zmm2, %zmm5
vfmadd231ps 128(%rdi,%rax), %zmm3, %zmm6
vfmadd231ps 192(%rdi,%rax), %zmm4, %zmm7
