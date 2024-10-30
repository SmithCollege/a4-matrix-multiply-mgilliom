#include <stdlib.h>
#include <stdio.h>

void MatrixMulOnHost(float* M, float* N, float* P, int Width) {
	for (int i = 0; i < Width; ++i)
		for (int j = 0; j < Width; ++j) {
		float sum = 0;
			for (int k = 0; k < Width; ++k) {
				float a = M[i * Width + k];
				float b = N[k * Width + j];
				sum += a * b;
			}
	P[i * Width + j] = sum;
	}
}

int main() {
  int size = 1000;

  float* x = malloc(sizeof(float) * size * size);
  float* y = malloc(sizeof(float) * size * size);
  float* z = malloc(sizeof(float) * size * size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      x[i * size + j] = 1; // x[i][j]
      y[i * size + j] = 1;
    }
  }

  MatrixMulOnHost(x, y, z, size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (z[i * size + j] != size) {
        printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
      }
    }
  }

  for (int i = 0; i < size; i++){
  	printf("%f\n", z[i]);
  }


  return 0;
}
