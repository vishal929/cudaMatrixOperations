
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
template <class T>

//scaling with generic type for matrices
__global__ void scaleMatrix(T* M, unsigned int maxDim,T scalar) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid;i < maxDim;i += blockDim.x * gridDim.x) {
        M[i] = scalar * M[i];
    }
}

//adding two matrices together
    //we are assuming here that proper checking has been done to ensure that M1 and M2 have the same size and so does the result matrix
template <class T>
__global__ void addMatrices(T* M1, T* M2, T* R, unsigned int maxDim) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i = tid;i < maxDim;i += blockDim.x * gridDim.x) {
       R[i] = M1[i] + M2[i];
   } 
}

//getting the transpose of a matrix, and storing result in another matrix 
    //M is the input, R is the result matrix with the same size (dimensions could be different, but R will have same # of entries)
template <class T>
__global__ void matrixTranspose(T* M, T* R, unsigned int rows, unsigned int cols) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i<rows*cols; i += blockDim.x * gridDim.x) {
        //need to put this entry in the right place in the result 
            //we have originally M[i] = M[rowIndex*(number columns) + colIndex]
            // in the transpose, rows becomes columns and vice versa
            // so our new place is M[colIndex(number rows) + rowindex]
                //to get rowindex from i, we just divide i by the number of columns and round down
        unsigned int convertedRow = (i % cols)*rows;
        unsigned int col = i / cols;
        R[convertedRow + col] = M[tid];

    }
    //result should be ok now for the transpose
}


//dot product between 2  vectors
    //v1 and v2 are input vectors of the same size
    //vr is a result allocation to store block results into
        //may need to reduce vr further after this kernel call
//we can use this to get vector magnitude also -> (v DOT v ) = mag(v) squared
template <class T>
__global__ void vectorDot(T* v1, T* v2, T* vr, unsigned int size) {
    //idea is to have a single thread handle a single component, and put the sum in the right component in shared memory
        //then in shared memory, we do add reduce and then have a thread put the result in global mem
        //if there needs to be more add reduce, then add reduce kernel can just be called on the result
    //for my implementation I will launch 256 threads per block 
    __shared__ T result[256];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < size; i += blockDim.x * gridDim.x) {
        //multiplying both components and placing them in the appropriate spot in the result vector
            //adding to result component does not matter because we are adding a bunch of results anyway
        result[threadIdx.x] += (v1[tid] * v2[tid]);
    }
    __syncthreads();
    //performing add reduce on shared memory and having a single thread write the result back to the results vector
        //may need to run further add reduce on the result
    for (unsigned int s = blockDim.x / 2;s > 0;s >>= 1) {
        if (tid < s) {
            //doing reduction with a neighbor halfway across the block
            result[tid] += result[tid + s];
        }
        //making sure threads are synced up within a block at the same reduce step
        __syncthreads();
    }

    //storing result for each block into global memory
    if (threadIdx.x == 0) {
        vr[blockIdx.x] = result[0];
    }

    //may need to add reduce again if the number of threads in a block is less than the number of components in the vector
}


//getting the cross product of 2 vectors, v1 and v2, and storing the result in vr
    //idea is simply to apply determinant to however many components we have, like in the 3D case
template <class T>
__global__ void vectorCross(T* v1, T* v2, T* vr, unsigned int size) {
    
}



int main()
{
    /*SCALING TEST*/
    /*
    int hostTestMatrix[100];
    int* cudaTestMatrix;
    cudaMalloc(&cudaTestMatrix, sizeof(int) * 100);
    for (int i = 0;i < 100;i++) {
        hostTestMatrix[i] = i;
    }
    //copying memory to device allocation
    cudaMemcpy(cudaTestMatrix, hostTestMatrix, 100 * sizeof(int), cudaMemcpyHostToDevice);
    //invoking kernel with a single thread for now
    scaleMatrix <<< 1, 1 >>> (cudaTestMatrix, 100, 2);
    //copying memory back to host
    cudaMemcpy(hostTestMatrix,cudaTestMatrix, 100 * sizeof(int), cudaMemcpyDeviceToHost);
    //printing out 10 rows of 10 columns
    for (int i = 0;i < 10;i++) {
        for (int j = 0;j < 10;j++) {
            printf("%d, ", hostTestMatrix[i * 10 + j]);
        }
        printf("\n");
    }*/

    /*ADDING TEST*/
    /*
    //allocating a matrix of negative values, result matrix should be all zeroes
    int hostTestMatrix[100];
    int* cudaTestMatrix;
    cudaMalloc(&cudaTestMatrix, sizeof(int) * 100);
    int hostTestSecondMatrix[100];
    int* cudaTestSecondMatrix;
    cudaMalloc(&cudaTestSecondMatrix, sizeof(int) * 100);

    for (int i = 0;i < 100;i++) {
        hostTestMatrix[i] = i;
        hostTestSecondMatrix[i] = -i;
    }

    //copying memory to device
    cudaMemcpy(cudaTestMatrix, hostTestMatrix, sizeof(int) * 100, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaTestSecondMatrix, hostTestSecondMatrix, sizeof(int) * 100, cudaMemcpyHostToDevice);

    //allocating memory for result
    int* resultMatrix;
    cudaMalloc(&resultMatrix, sizeof(int) * 100);

    //calling matrix add
    addMatrices << <4, 256 >> > (cudaTestMatrix, cudaTestSecondMatrix, resultMatrix, 100);

    //copying result to first matrix to print out
    cudaMemcpy(hostTestMatrix, resultMatrix, sizeof(int) * 100, cudaMemcpyDeviceToHost);
    
    //printing result
    for (int i = 0;i < 10;i++) {
        for (int j = 0;j < 10;j++) {
            printf("%d, ", hostTestMatrix[i * 10 + j]);
        }
        printf("\n");
    }*/

    /*Vector Dot Product Test*/
    int firstHostVector[10];
    int secondHostVector[10];
    for (int i = 0;i < 10;i++) {
        //first host ={1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        // second host = { 0-9}
        firstHostVector[i] = i + 1;
        secondHostVector[i] = i;
    }

    int* firstDeviceVector;
    int* secondDeviceVector;
    int* resultAllocation;
    cudaMalloc(&firstDeviceVector, sizeof(int) * 10);
    cudaMalloc(&secondDeviceVector, sizeof(int) * 10);
    cudaMalloc(&resultAllocation, sizeof(int) * 10);

    return 0;
}

