#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define TILE_WIDTH 16


__global__ void MatrixMulOnDevice(float* M, float* N, float* P, int Width) {
	__shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	// Identify the row and column of the P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
	// Loop over the M and N tiles required to compute the P element
	// The code assumes that the Width is a multiple of TILE_WIDTH!
	for (int m = 0; m < Width/TILE_WIDTH; ++m) {
		// Collaborative loading of M and N tiles into shared memory
		subTileM[ty][tx] = M[Row*Width + m*TILE_WIDTH+tx];
		subTileN[ty][tx] = N[(m*TILE_WIDTH+ty)*Width+Col];
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k){
			Pvalue += subTileM[ty][k] * subTileN[k][tx];
			__syncthreads();
		}
	}
	P[Row*Width+Col] = Pvalue;
}



double get_clock(){
	struct timeval tv; int ok;
	ok = gettimeofday(&tv, (void *) 0);
	if (ok<0) { printf("gettimeofday error"); }
	return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}



int main(void) {
  int size; //has to be a multiple of tile_width^2
  printf("Width of P (multiple of 256): ");
  scanf("%d", &size);

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

  dim3 dimGrid(ceil((1.0*size)/TILE_WIDTH), ceil((1.0*size)/TILE_WIDTH), 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  printf("width %d: dimGrid %d, dimBlock %d \n", size, dimGrid.x, dimBlock.x);

  double t0 = get_clock();
  MatrixMulOnDevice<<<dimGrid, dimBlock>>>(M, N, P, size);
  cudaMemcpy(z, P, sizeof(float)*size*size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double t1 = get_clock();

  printf("time: %f s \n", (t1-t0));
  
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
