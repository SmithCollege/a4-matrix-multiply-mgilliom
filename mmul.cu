#include <stdlib.h>
#include <stdio.h>

#define BLOCKSIZE 256


__global__ void MatrixMulOnDevice(float* M, float* N, float* P, int Width) {
	int gindex = threadIdx.x + blockIdx.x*blockDim.x;
	
    if (gindex < Width*Width){
    	int row = gindex/Width;
    	int col = gindex%Width;
    	float sum = 0;
		for (int i = 0; i < Width; ++i){
			float a = M[row * Width + i];
			float b = N[i * Width + col];
			sum += a * b;
			P[gindex] = sum;
		}
	}
}

int main(void) {
  int size = 1000;

  float *x, *y, *z, *M, *N, *P;

  x = (float*)malloc(sizeof(float) * 1000 * 1000);
  y = (float*)malloc(sizeof(float) * size * size);
  z = (float*)malloc(sizeof(float) * size * size);

  cudaMalloc(&M, sizeof(float)*size*size);
  cudaMalloc(&N, sizeof(float)*size*size);
  cudaMalloc(&P, sizeof(float)*size*size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      x[i * size + j] = 1; // x[i][j]
      y[i * size + j] = 1;
    }
  }

  cudaMemcpy(M, x, sizeof(float)*size*size, cudaMemcpyHostToDevice);
  cudaMemcpy(N, y, sizeof(float)*size*size, cudaMemcpyHostToDevice);
  cudaMemcpy(P, z, sizeof(float)*size*size, cudaMemcpyHostToDevice);

  int numThreads = size * size;
  int numBlocks = ceil(1.0 * numThreads / BLOCKSIZE);
  printf("numthreads %d, numblocks %d \n", numThreads, numBlocks);
  MatrixMulOnDevice<<<numBlocks, BLOCKSIZE>>>(M, N, P, size);

  cudaMemcpy(z, P, sizeof(float)*size*size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (z[i * size + j] != size) {
        printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
      }
    }
  }

  #if 0
  for (int i = 0; i < size; i++){
  	printf("%f\n", z[i]);
  }
  #endif

  cudaFree(M);
  cudaFree(N);
  cudaFree(P);
  free(x);
  free(y);
  free(z);
  

  return 0;
}
