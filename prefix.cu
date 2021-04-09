#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>

__global__ void short_prefix_sum(int *array, unsigned length, unsigned step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < length) && (i >= step)) {
        array[i] += array[i - step];
    }
    __syncthreads();
}

void short_prefix(int *host_array, unsigned length) {
    int array_size = length * sizeof(int);
    int *device_array; cudaMalloc((void **) &device_array, array_size);
    cudaMemcpy(device_array, host_array, array_size, cudaMemcpyHostToDevice);
    
    dim3 numBlocks( length );
    dim3 threadsPerBlock(1);
    for (unsigned step=1; step<length; step<<=1) {
        short_prefix_sum<<<numBlocks, threadsPerBlock>>>(device_array, length, step);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(host_array, device_array, array_size, cudaMemcpyDeviceToHost);
    cudaFree(device_array);
}

__global__ void long_prefix_upsweep(int *array, unsigned length, unsigned d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = (int) pow(2, d+1);
    if ((i < length) && (0 == (i % p))) {
        array[i + p - 1] += array[i + p/2 - 1];
    }
    __syncthreads();
}

__global__ void long_prefix_downsweep(int *array, unsigned length, unsigned d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = (int) pow(2, d+1);
    if ((i < length) && (0 == (i % p))) {
        int tmp = array[i + p/2 - 1];
        array[i + p/2 - 1] = array[i + p - 1];
        array[i + p - 1] = tmp + array[i + p/2 - 1];
    }
    __syncthreads();
}

void long_prefix(int *host_array, unsigned length) {
    int array_size = length * sizeof(int);
    int *device_array; cudaMalloc((void **) &device_array, array_size);
    cudaMemcpy(device_array, host_array, array_size, cudaMemcpyHostToDevice);
    
    dim3 numBlocks(1);
    dim3 threadsPerBlock(1024);
    int  l = log(length)/log(2);
    for (int d=0; d < l; d++) {
        long_prefix_upsweep<<<numBlocks, threadsPerBlock>>>(device_array, length, d);
        cudaDeviceSynchronize();
    }
    for (int d=l; d >= 0; d--) {
        long_prefix_downsweep<<<numBlocks, threadsPerBlock>>>(device_array, length, d);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(host_array, device_array, array_size, cudaMemcpyDeviceToHost);
    cudaFree(device_array);
}

int main(int argc, char **argv){
    int length;
    if (2 == argc) {
        errno = 0;
        length = strtol(argv[1], NULL, 10);
    }
    else {
        fprintf(stderr, "argument must be exactly one integer\n");
        return 22;
    }
    if (errno) {
        fprintf(stderr, "error %d: %s\n", errno, strerror(errno));
        return errno;
    }
    int array_size = length * sizeof(int);
    int *host_array = (int*) malloc(array_size);
    for (int i=0; i<length; ++i) {
        host_array[i] = 1;
    }
    
    long_prefix(host_array, length);
    
    for (int i=0; i<length; ++i) {
        printf("%d\n", host_array[i]);
    }
    
    free(host_array);
}

