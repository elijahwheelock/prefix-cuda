#include <stdio.h>
#include <math.h>
#include <unistd.h>

__global__ void short_prefix_sum(int *array, unsigned length, unsigned step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < length) && (i >= step)) {
        array[i] += array[i - step];
    }
    __syncthreads();
}

void short_prefix(int *device_array, unsigned length) {
    dim3 numBlocks(1);
    dim3 threadsPerBlock(1024);
    for (unsigned step=1; step<length; step<<=1) {
        short_prefix_sum<<<numBlocks, threadsPerBlock>>>(device_array, length, step);
        cudaDeviceSynchronize();
    }
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

void long_prefix(int *device_array, unsigned n) {
    dim3 numBlocks(1);
    dim3 threadsPerBlock(1024);
    int  l = log(n)/log(2)
    for (int d=0; d < l; d++) {
        long_prefix_upsweep<<<numBlocks, threadsPerBlock>>>(device_array, n, d);
        cudaDeviceSynchronize();
    }
    for (int d=l; d>=0; d--) {
        long_prefix_downsweep<<<numBlocks, threadsPerBlock>>>(device_array, length, d);
        cudaDeviceSynchronize();
    }
}

int main(){
    int n = 16;
    int array_size = n * sizeof(int);
    int *host_array = (int*) malloc(array_size);
    int *device_array; cudaMalloc((void **) &device_array, array_size);
    for (int i=0; i<n; ++i) { host_array[i] = 1; }
    cudaMemcpy(device_array, host_array, array_size, cudaMemcpyHostToDevice);
    long_prefix(device_array, n);
    cudaMemcpy(host_array, device_array, array_size, cudaMemcpyDeviceToHost);
    for (int i=0; i<n; i++) {
        printf("%d\n", host_array[i]);
    }

    free(host_array);
    cudaFree(device_array);
}

