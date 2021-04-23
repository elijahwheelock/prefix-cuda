#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#define log2(n) log(n)/log(2)

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
    int p = 1 << (d+1);
    if ((i < length) && (0 == (i % p))) {
        array[i + p - 1] += array[i + p/2 - 1];
    }
    __syncthreads();
}

__global__ void long_prefix_downsweep(int *array, unsigned length, unsigned d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = 1 << (d+1);
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
    
    dim3 numBlocks(length / 1024);
    dim3 threadsPerBlock(1024);
    int  l = log2(length);
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
    int input_n;
    if (2 == argc) {
        errno = 0;
        input_n = strtol(argv[1], NULL, 10);
    }
    else {
        fprintf(stderr, "argument must be exactly one integer\n");
        return 22;
    }
    if (errno) {
        fprintf(stderr, "error %d: %s\n", errno, strerror(errno));
        return errno;
    }
    int length = 1 << input_n;
    printf("length: %d\n", length);
    
    int array_size = length * sizeof(int);
    int *host_array = (int*) malloc(array_size);
    for (int i=0; i<length; ++i) {
        host_array[i] = 1;
    }
    
    long_prefix(host_array, length);
    
    bool not_expected = false;
    for (int i=0; i<length; ++i) {
        if (i != host_array[i]) {
            not_expected = true;
            break;
        }
    }
    if (not_expected) {
        for (int i=0; i<length; ++i) {
            printf("%d ", host_array[i]);
        }
        printf("failure!\n");
    } else { printf("success!\n"); }
    
    free(host_array);
}

