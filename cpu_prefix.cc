#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#define LOG2(n) log(n)/log(2)

void print_array(int *array, unsigned length) {
    for (int i=0; i<length; i++) {
        printf("%d ", array[i]);
    }
    printf("\n\n");
}

void prefix_upsweep(int *array, unsigned length, unsigned stride) {
    for (unsigned i=0; (i+stride)<=length; i+=stride) {
        array[i + stride - 1] += array[i + stride/2 - 1];
    }
}

void prefix_downsweep(int *array, unsigned length, unsigned stride) {
    for (unsigned i=0; 0<(i+stride/2) && (i+stride)<=length; i+=stride) {
      //printf("%d %d -- %d: %d, %d: %d\n", i, stride, i+stride/2-1, array[i+stride/2], i+stride-1, array[i+stride-1]);
        int tmp = array[i+stride/2-1];
        array[i+stride/2-1] = array[i+stride  -1];
        array[i+stride  -1] = array[i+stride/2-1] + tmp;
    }
}

void prefix(int *host_array, unsigned length) {
    int array_size = length * sizeof(int);
    int *device_array; 
    assert(NULL != (device_array = (int *) malloc(array_size)));
    memcpy(device_array, host_array, array_size);
     
    unsigned stride = 2;
    for (; stride<=length; stride<<=1) {
        prefix_upsweep(device_array, length, stride);
    }
    device_array[length-1] = 0;
  //print_array(device_array, length);
    for (stride>>=1; stride >= 1; stride>>=1) {
        prefix_downsweep(device_array, length, stride);
    }
     
    memcpy(host_array, device_array, array_size);
    free(device_array);
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
    //printf("length: %d\n", length);
     
    int array_size = length * sizeof(int);
    int *host_array;
    assert(NULL != (host_array = (int*) malloc(array_size)));
     
    for (int i=0; i<length; ++i) {
        host_array[i] = 1;
    }
  //print_array(host_array, length);
     
    prefix(host_array, length);
     
    bool not_expected = false;
    for (int i=0; i<length; ++i) {
        if (i != host_array[i]) {
            not_expected = true;
            break;
        }
    }
    if (not_expected) {
        print_array(host_array, length);
        printf("\nfailure!\n");
    } else { printf("\nsuccess!\n"); }
     
    free(host_array);
}

