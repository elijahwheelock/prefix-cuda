#include <stdio.h>  // printf
#include <errno.h>  // errno
#include <assert.h> // assert
#include <stdint.h> // uint32_t
#include <time.h>   // performance testing

#ifdef synchronize
    #define DEVICE_SYNC() __syncthreads()
    #define HOST_SYNC()   cudaDeviceSynchronize()
#else
    #define DEVICE_SYNC() 
    #define HOST_SYNC()
#endif

void print_array(int *array, uint32_t length) {
    for (int i=0; i<length; i++) {
        printf("%d ", array[i]);
    }
    printf("\n\n");
}

__global__ void prefix_upsweep(int *array, uint32_t length, uint32_t stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i *= stride;
    array[i + stride - 1] += array[i + stride/2 - 1];
    DEVICE_SYNC();
}

__global__ void zero_last_element(int *array, uint32_t length) {
    array[length-1] = 0;
}

__global__ void prefix_downsweep(int *array, uint32_t length, uint32_t stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i *= stride;
    int tmp = array[i+stride/2-1];
    array[i+stride/2-1] = array[i+stride  -1];
    array[i+stride  -1] = array[i+stride/2-1] + tmp;
    DEVICE_SYNC();
}

void prefix(int *host_array, uint32_t length) {
    int array_size = length * sizeof(int);
    int *device_array; 
    cudaError err;
    err = cudaMallocManaged((void **) &device_array, array_size, cudaMemAttachHost);
    if (err) {
        printf("cudaMalloc failed with error %d\n", err); 
    }
    assert(cudaSuccess == err);
    assert(cudaSuccess == cudaMemcpy(device_array, host_array, array_size, cudaMemcpyHostToDevice));
    
    uint32_t stride = 2;
    for (; stride<=length; stride<<=1) {
      //printf("stride: %d\n", stride);
        dim3 numBlocks((length / stride));
        dim3 threadsPerBlock(1);
        prefix_upsweep<<<numBlocks, threadsPerBlock>>>(device_array, length, stride);
        HOST_SYNC();
    }
    zero_last_element<<<dim3(1), dim3(1)>>>(device_array, length);
    for (stride>>=1; stride > 1; stride>>=1) {
      //printf("stride: %d\n", stride);
        dim3 numBlocks((length / stride));
        dim3 threadsPerBlock(1);
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
    uint32_t length = 1 << input_n;
  //printf("length: %d\n", length);
  //size_t gpu_free_mem, gpu_total_mem;
  //cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
  //printf("GPU memory free: %d bytes; total: %d bytes\n", gpu_free_mem, gpu_total_mem);
    
    unsigned trials=10;
    double times[trials];
    uint32_t array_size = length * sizeof(int);
    for (int j=0; j<trials; j++) {
        int *host_array = (int*) malloc(array_size);
        assert(NULL != host_array);
        
        for (int i=0; i<length; ++i) {
            host_array[i] = 1;
        }
        
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        prefix(host_array, length);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        
        double elapsed_time = ((double) (t1.tv_sec - t0.tv_sec)) + ((double) (t1.tv_nsec - t0.tv_nsec))/1000000000;
        times[j] = elapsed_time;
        
        bool not_expected = false;
        for (int i=0; i<length; ++i) {
            if (i != host_array[i]) {
                printf("expected %d at index %d, found %d\n", i, i, host_array[i]);
                not_expected = true;
                break;
            }
        }
        assert(!not_expected);
      //if (not_expected) {
      //    //print_array(host_array, length);
      //    printf("failure!\n");
      //} else {
      //    printf("success!\n");
      //}
        
        free(host_array);
    }
    double min_time = 10000;
    for (int i=0; i<trials; i++) {
        if (times[i] < min_time) { min_time = times[i]; }
    }
    printf("%f\n", min_time);
}

