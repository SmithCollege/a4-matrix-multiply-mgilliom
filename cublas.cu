#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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


double get_clock(){
	struct timeval tv; int ok;
	ok = gettimeofday(&tv, (void *) 0);
	if (ok<0) { printf("gettimeofday error"); }
	return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


int main(void) {
  int size; 
  printf("Width of P: ");
  scanf("%d", &size);

   const float alpha = 1.0f;
   const float beta  = 0.0f;

  float *x, *y, *z, *M, *N, *P;

  x = (float*)malloc(sizeof(float) * size * size);
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

  double t0 = get_clock();
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, N, size, M, size, &beta, P, size);
  cudaMemcpy(z, P, sizeof(float)*size*size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double t1 = get_clock();
  printf("time: %f s\n", (t1-t0));
  
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
