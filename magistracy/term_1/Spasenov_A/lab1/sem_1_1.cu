
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <string>
#include <stdio.h>

static const int DIM   = 128;
static const int NODES = DIM * DIM;
static const double L = 1.0 * NODES;
static const double TIME_OVERALL = 30.0; // seconds
static const double DIRICHLET = 0.0;
static const double NEUMAN = 5;

__global__ void Kernel(double *v, double *vPrev, double hx, double ht, unsigned long timeSteps) {
	unsigned int idx_X = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int idx_Y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx   = idx_X + idx_Y * DIM;

    for (int iStep = 0; iStep < timeSteps; ++iStep) {
        if(idx > 0 && idx < (NODES - 1))
	        v[idx] = (((vPrev[idx + 1] - 2 * vPrev[idx] + vPrev[idx - 1]) * ht) / (hx * hx)) + vPrev[idx];

	    __syncthreads();

        if (idx == 0) {
            v[idx] = DIRICHLET;
        } else if (idx == (NODES - 1)) {
            v[idx] = hx * NEUMAN + v[idx - 1];
        }

        vPrev[idx] = v[idx];

        __syncthreads();
    }
}

int main()
{
	size_t mem_size = sizeof(double) * NODES;
	cudaError cudaStatus;

	double hx = L / (NODES - 1);
	double ht = 1e-1;//(hx * hx) / 2;

    unsigned long timeSteps = TIME_OVERALL / ht;

    std::cout << "hx = " << hx << std::endl;
    std::cout << "ht = " << ht << std::endl;
    std::cout << "steps = " << timeSteps << std::endl;

    if (hx < 1e-9 || ht < 1e-14) {
        std::cout << "too small values of hx and (or) ht: " << std::endl;
        return 1;
    }

	double *devNodes;
    double *prevDevNodes;
    cudaMalloc((void **) &devNodes, mem_size);
    cudaMalloc((void **) &prevDevNodes, mem_size);
	cudaMemset(prevDevNodes, 0, mem_size);
    cudaMemset(devNodes, 0, mem_size);

	dim3 N_Grid(DIM / 32, DIM / 32, 1);
	dim3 N_Block(32, 32, 1);

	cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess) {
		std::cout << "failed to set cuda device 0" << std::endl;
		cudaFree(devNodes);
		return 1;
	}

	Kernel <<< N_Grid, N_Block >>> (devNodes, prevDevNodes, hx, ht, timeSteps);
	cudaStatus = cudaGetLastError();

	if(cudaStatus != cudaSuccess) {
        std::cout << "last error: " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaFree(devNodes);
		return 1;
	}

    
	double *hostNodes = (double *) malloc(mem_size);
	cudaMemcpy(hostNodes, devNodes, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(devNodes);
    cudaFree(prevDevNodes);

	for(size_t i = 0; i < NODES; i++) {
        std::cout << i << ": " << hostNodes[i] << std::endl;
	}

	free(hostNodes);
	return 0;
}
