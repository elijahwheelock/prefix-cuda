#include <stdio.h>
#include <errno.h>  // errno
#include <assert.h> // assert
#include <stdint.h> // uint64_t
#ifdef synchronize
    #define DEVICE_SYNC() __syncthreads()
    #define HOST_SYNC()   cudaDeviceSynchronize()
#else
    #define DEVICE_SYNC() 
    #define HOST_SYNC()
#endif

void print_array(int *array, uint64_t length) {
    for (int i=0; i<length; i++) {
        printf("%d ", array[i]);
    }
    printf("\n\n");
}

__global__ void prefix_upsweep(int *array, uint64_t length, uint64_t stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i *= stride;
    if (0<(i+stride/2) && (i+stride)<=length) {
        array[i + stride - 1] += array[i + stride/2 - 1];
    }
    DEVICE_SYNC();
}

__global__ void zero_last_element(int *array, uint64_t length) {
    array[length-1] = 0;
}

__global__ void prefix_downsweep(int *array, uint64_t length, uint64_t stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i *= stride;
    if ((0<(i+stride/2)) && ((i+stride-1)<length)) {
        int tmp = array[i+stride/2-1];
        array[i+stride/2-1] = array[i+stride  -1];
        array[i+stride  -1] = array[i+stride/2-1] + tmp;
    }
    DEVICE_SYNC();
}

void prefix(int *host_array, uint64_t length) {
    int array_size = length * sizeof(int);
    int *device_array; 
    assert(cudaSuccess == cudaMalloc((void **) &device_array, array_size));
    assert(cudaSuccess == cudaMemcpy(device_array, host_array, array_size, cudaMemcpyHostToDevice));
    
    dim3 numBlocks;
    dim3 threadsPerBlock;
    uint64_t stride = 2;
    for (; stride<=length; stride<<=1) {
      //printf("stride: %d\n", stride);
        dim3 numBlocks(length / stride);
        dim3 threadsPerBlock(1);
      //numBlocks = dim3(1);
      //threadsPerBlock = dim3(1024);
        prefix_upsweep<<<numBlocks, threadsPerBlock>>>(device_array, length, stride);
        HOST_SYNC();
    }
    zero_last_element<<<dim3(1), dim3(1)>>>(device_array, length);
    for (stride>>=1; stride > 1; stride>>=1) {
      //printf("stride: %d\n", stride);
        numBlocks = dim3(length / stride);
        threadsPerBlock = dim3(1);
        prefix_downsweep<<<numBlocks, threadsPerBlock>>>(device_array, length, stride);
        HOST_SYNC();
    }
    
    assert(cudaSuccess == cudaMemcpy(host_array, device_array, array_size, cudaMemcpyDeviceToHost));
  //print_array(host_array, length);
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
    uint64_t length = 1 << input_n;
  //printf("length: %d\n", length);
    
    uint64_t array_size = length * sizeof(int);
    int *host_array = (int*) malloc(array_size);
    assert(NULL != host_array);
    
    for (int i=0; i<length; ++i) {
        host_array[i] = 1;
    }
    
    prefix(host_array, length);
    
    bool not_expected = false;
    for (int i=0; i<length; ++i) {
        if (i != host_array[i]) {
            printf("expected %d at index %d, found %d\n", i, i, host_array[i]);
            not_expected = true;
            break;
        }
    }
    if (not_expected) {
        //print_array(host_array, length);
        printf("failure!\n");
    } else { printf("success!\n"); }
    
    free(host_array);
}

