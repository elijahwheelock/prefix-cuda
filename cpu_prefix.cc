#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#define log2(n) log(n)/log(2)

void prefix_upsweep(int *array, unsigned length, unsigned d) {
    for (unsigned i=0; i<length; ++i) {
        unsigned p = 1 << (d+1);
        if ((i < length) && (0 == (i % p))) {
            array[i + p - 1] += array[i + p/2 - 1];
        }
    }
}

void prefix_downsweep(int *array, unsigned length, unsigned d) {
    for (unsigned i=0; i<length; ++i) {
        unsigned p = 1 << (d+1);
        if ((i < length) && (0 == (i % p)) && ((i+p) < length)) {
            int tmp = array[i + p/2 - 1];
            array[i+p/2-1] = array[i+p-1];
            array[i+p-1] = tmp + array[i+p/2-1];
        }
    }
}

void prefix(int *host_array, unsigned length) {
    int array_size = length * sizeof(int);
    int *device_array; 
    assert(NULL != (device_array = (int *) malloc(array_size)));
    memcpy(device_array, host_array, array_size);
    
    int  l = log2(length);
    for (int d=0; d < l; d++) {
        prefix_upsweep(device_array, length, d);
    }
    for (int d=l; d >= 0; d--) {
        prefix_downsweep(device_array, length, d);
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
    
    prefix(host_array, length);
    
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

