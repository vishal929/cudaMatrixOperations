
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


//c is the result array, b and a are the input arrays
__global__ void myArrayAdd(int *c, int *b, int* a, unsigned int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    for (int i = tid;i < size;i+=blockDim.x*gridDim.x) {
        c[i] = b[i] + a[i];
        printf("added on index %d\n", i);
    }
}

int main()
{
    const int arraySize = 5;
    //int a[arraySize];
    int* a;
    int* b;
    cudaMallocManaged(&a, arraySize * sizeof(int));
    cudaMallocManaged(&b, arraySize * sizeof(int));
    //int b[arraySize];
    
    //int c[arraySize] = { 0 };
    int* c;
    cudaMallocManaged(&c, arraySize * sizeof(int));

    for (int i = 0;i < arraySize;i++) {
        a[i] = i;
        b[i] = 10 * i;
        c[i] = 0;
    }
    

    // Add vectors in parallel.
    int blockSize = 256;
    int numBlocks ;
    if (arraySize % blockSize == 0) {
        //then array size is divisible by our block size
        numBlocks = arraySize / blockSize;
    }
    else {
        //then array size is not divisible by our block size
        numBlocks = (arraySize / blockSize) + 1;
    }
    printf("NUM BLOCKS: %d\n", numBlocks);
    
    myArrayAdd<<<numBlocks,blockSize>>>(c, a, b,arraySize);
    cudaDeviceSynchronize();
    printf("PRINTING RESULTS!\n");
    for (int i = 0;i < arraySize;i++) {
        printf("%d, ", c[i]);
    }
    printf("\n");

    return 0;
}


