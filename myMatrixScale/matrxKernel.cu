#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <stdio.h>
#include <stdlib.h>


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
    //final result to copy back into global memory in this blocks index at the end
    T finalResult=0;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < size; i += blockDim.x * gridDim.x) {
        //multiplying both components and placing them in the appropriate spot in the result vector
            //adding to result component does not matter because we are adding a bunch of results anyway
        result[threadIdx.x] += (v1[tid] * v2[tid]);
        //finalResult += (v1[tid] * v2[tid]);

        //syncing the results in this batch
        __syncthreads();
        //performing add reduce on shared memory (so only threadid matters here)
        for (unsigned int s = blockDim.x / 2;s > 0;s >>= 1) {
            if (threadIdx.x < s) {
                result[threadIdx.x] += result[threadIdx.x + s];
            }
            //making sure every thread in this level of reduction has finished adding
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            //updating the finalresult value
            finalResult += result[0];
        }
        //need syncthreads here to make sure no threads change the result value before finalResult can be updated
        __syncthreads();
        
    }

    //copying into global memory with atomic add
    if (threadIdx.x == 0) {
        atomicAdd(vr, &finalResult);
        //vr[blockIdx.x] = finalResult;
    }
    //may need to add reduce again if the number of threads in a block is less than the number of components in the vector
}


//getting the cross product of 2 vectors, v1 and v2, and storing the result in vr
    //idea is simply to apply determinant to however many components we have, like in the 3D case
template <class T>
__global__ void vectorCross(T* v1, T* v2, T* vr, unsigned int size) {
    
}

//scaling with generic type for matrices
template <class T>
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

//matrix multiplication between 2 matrices M1 and M2; result is stored in R
    //row1 and col1 are dimensions of M1, row2 and col2 are dimensions of M2
    //recall that a pxq matrix multiplied by qxm will have result pxm
        //so result should have size row1 x col2
    //allocation of the result is assumed to then be correct by the user 
//DISCLAIMER: my idea here is that each block of threads handles a vector dot product
            //a lot of solutions I saw use 1 thread to handle each dot product, but I would like to exploit as much parallelism as possible
                   //As a result, I want one block of threads to handle each individual dot product
            //I will try and compare both solutions in terms of throughput on large matrices
template<class T>
__global__ void matrixMultiply(T* M1, T* M2, T* R, unsigned int row1, unsigned int col1, unsigned int row2, unsigned int col2) {
    //need to reduce shared memory in place and then update the corresponding global value in R
    //we will launch the kernel with 256 threads per block
        //I will launch as many blocks as components in the result vector to fully exploit parallelism
    __shared__ T result[256];
    //finalSum for thread 0 in the block to update 
    T finalResult = 0;

    
    //each block of threads handles calculation of 1 entry in the output  
        //this is opposed to a lot of implementations where a single thread handles calculation of 1 entry in the output
        //hopefully this means really fast computation times, we will see
        //so the structure is that our grid contains a 2d array of blocks
            //each block has a 1D array of threads
     
    /*IDEA BELOW TO HAVE BLOCKS HANDLE INDIVIDUAL DOT PRODUCTS TO EXPLOIT MAXIMUM PARALLELISM*/
    
    for (int i = blockIdx.x;i < row1;i += gridDim.x) {
        for (int j = blockIdx.y;j < col2;j += gridDim.y) {
            //doing vector dot for this particular block
            //int tid = gridIdx.x * blockIdx.x + threadIdx.x;
            
            for (int z = threadIdx.x;z< col1;z += blockDim.x) {
                result[threadIdx.x] = M1[(i*col1)+z] * M2[(z*col2)+j];
                if (i == 0 && j==0) {
               //debugging for the first entry 
                printf("thread: %d has computed value %d!\n", threadIdx.x,result[threadIdx.x]);

                }

                __syncthreads();

                //doing parallel add reduce for the vector sum
                for (unsigned int s = blockDim.x / 2;s > 0;s >>= 1) {
					if (threadIdx.x < s) {
						result[threadIdx.x] += result[threadIdx.x + s];
					}
					//making sure every thread in this level of reduction has finished adding
					__syncthreads();
			    }

                //now the first entry has the total sum
                if (threadIdx.x == 0) {
					//updating the finalresult value
					finalResult += result[0];
                }
                //need syncthreads here to make sure no threads change the result value before finalResult can be updated
                __syncthreads();
            }
            //now we can update the global value of the matrix entry
            if (threadIdx.x == 0) {
                R[(i * col2) + j] = finalResult;
                finalResult = 0;
            }
            //making sure finalResult isnt modified by some other thread in the block before it can be updated to global memory
            __syncthreads();
        }
    }
    
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
    /*
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

    //copying host memory to device
    cudaMemcpy(firstDeviceVector, firstHostVector, sizeof(int) * 10, cudaMemcpyHostToDevice);
    cudaMemcpy(secondDeviceVector, secondHostVector, sizeof(int) * 10, cudaMemcpyHostToDevice);

    //running kernel
    vectorDot << <1, 256 >> > (firstDeviceVector, secondDeviceVector, resultAllocation, 10);

    //copying result allocation to buffer to print
    int result;
    cudaMemcpy(&result, resultAllocation, sizeof(int), cudaMemcpyDeviceToHost);
    printf("RESULT %d\n", result); */

    /*Matrix Transpose Test*/
    /*
    //we will suppose that hostMatrix is 5x20 matrix, so the transpose should be 20x5
    int hostMatrix[100];
    for (int i = 0;i < 100;i++) {
        hostMatrix[i] = i;
    }
    int* cudaMatrix;
    int* cudaResultMatrix;
    cudaMalloc(&cudaMatrix, sizeof(int) * 100);
    cudaMalloc(&cudaResultMatrix, sizeof(int) * 100);
    cudaMemcpy(cudaMatrix, hostMatrix, sizeof(int) * 100, cudaMemcpyHostToDevice);
    //calling kernel
    matrixTranspose << <1, 256 >> > (cudaMatrix, cudaResultMatrix, 5, 20);
    //copying result to host allocation and printing
    cudaMemcpy(hostMatrix, cudaResultMatrix, sizeof(int) * 100, cudaMemcpyDeviceToHost);
    //printing result
    for (int i = 0;i < 20;i++) {
        for (int j = 0;j < 5;j++) {
            printf("%d, ", hostMatrix[i * 5 + j]);
        }
        printf("\n");
    }*/

    /*TESTING MULTIPLICATION OF MATRICES*/
    //two 10x10 matrices
    int firstMatrix[100];
    int secondMatrix[100];
    for (int i = 0;i < 100;i++) {
        firstMatrix[i] = i;
        secondMatrix[i] = i;
    }

    //cuda copying
    int* firstMatrixDevice;
    int* secondMatrixDevice;
    int* resultMatrix;
    cudaMalloc(&firstMatrixDevice, sizeof(int) * 100);
    cudaMalloc(&secondMatrixDevice, sizeof(int) * 100);
    cudaMalloc(&resultMatrix, sizeof(int) * 100);

    //copying
    cudaMemcpy(firstMatrixDevice, firstMatrix, sizeof(int) * 100, cudaMemcpyHostToDevice);
    cudaMemcpy(secondMatrixDevice, secondMatrix, sizeof(int) * 100, cudaMemcpyHostToDevice); 

    //getting 2D grid of blocks, which are 1D arrays of threads
        //block for each output
    dim3 gridDim = dim3(10, 10);
        //1D array threads for iteration
    dim3 blockDim = dim3(256, 1);

    //calling the kernel
    matrixMultiply << <gridDim, blockDim >> > (firstMatrixDevice, secondMatrixDevice, resultMatrix, 10, 10, 10, 10);
    
    //copying results to printable array
    cudaMemcpy(firstMatrix, resultMatrix, sizeof(int) * 100, cudaMemcpyDeviceToHost);

    for (int i = 0;i < 10;i++) {
        for (int j = 0;j < 10;j++) {
            printf("%d, ", firstMatrix[(i * 10) + j]);
        }
        printf("\n");
    }

    return 0;
}

