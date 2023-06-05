#include <cuda_runtime.h>
#include <stdio.h>
#include <armadillo>
#include "helper/cuda_helper.cu"

__global__ void fun(float *i, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N)printf("%f\n", i[index]);
}

int main() {
    arma::fmat A(5, 5, arma::fill::ones);
    float *a;
    cudaMalloc(&a, sizeof(float) * A.size());
    cudaMemcpy(a, A.memptr(), sizeof(float) * A.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    fun<<<1, 25>>>(a, 25);
    cudaDeviceSynchronize();
    return 0;
}
