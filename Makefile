SYNC=-Dsynchronize
all: gpu_prefix cpu_prefix

gpu_prefix: prefix.cu
	nvcc -g prefix.cu -o gpu_prefix.fatbin $(SYNC)

cpu_prefix: cpu_prefix.c
	gcc -g -O3 cpu_prefix.c -o cpu_prefix.o
