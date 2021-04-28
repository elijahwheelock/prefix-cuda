all: gpu_prefix cpu_prefix

gpu_prefix: prefix.cu
	nvcc prefix.cu -o gpu_prefix.fatbin -Dsynchronize

cpu_prefix: cpu_prefix.cc
	g++ -g cpu_prefix.cc -o cpu_prefix.o
