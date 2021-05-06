
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

#define MAX_SAVES 1000
#define QTD_NUMBERS 3

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void GenSeq(uint32_t *divs, uint8_t*arr, uint32_t arrSize, uint32_t* index, uint32_t MAXgTid){
    uint32_t ThrPerBlk = blockDim.x;
	uint32_t MYbid = blockIdx.x;
	uint32_t MYtid = threadIdx.x;
    uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;
   
    uint8_t i;
    // uint32_t qtdNumbers = sizeof(divs) - 1;
    //uint8_t tempArr[qtd_numbers];
    uint32_t offset = MYgtid * QTD_NUMBERS;
    // printf("ThrPerBlk: %u, Mybid: %u, Mytid: %d, Mygtid: %u, offset: %u\n", ThrPerBlk, MYbid, MYtid, ThrPerBlk * MYbid + MYtid, offset);
    if(offset >= MAXgTid)
        return;
    
    uint8_t tempArr[QTD_NUMBERS];

    for(i = 0; i < QTD_NUMBERS; i++){
        tempArr[i] = (MYgtid %  divs[i]) / divs[i + 1];
    }

    uint32_t currIndex = atomicAdd(index, QTD_NUMBERS);
    if(currIndex > MAX_SAVES)
        return;

    printf("ThrPerBlk: %u, Mybid: %u, Mytid: %d, Mygtid: %u, offset: %u, currIndex: %u\n", ThrPerBlk, MYbid, MYtid, MYgtid, offset, currIndex);
    for(i = 0; i < QTD_NUMBERS; i++){
        arr[currIndex] = tempArr[i];
        currIndex++;
    }
}   

int main(){
	cudaError_t cudaStatus;
	cudaEvent_t time1, time2, time3;
	float totalTime, kernelExecutionTime, tfrGPUtoCPU;
    uint32_t ThrPerBlk = 32, NumBlocks = 0;

	int NumGPUs = 0; cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		std::cout << "\nNo CUDA Device is available\n";
		exit(EXIT_FAILURE);
	}

    uint32_t i = 0;
    uint32_t possibilities = 1;
    uint8_t  numbers[] = {3,3,3};
    // uint8_t  qtd_numbers = sizeof(numbers);
    for(i = 0; i < QTD_NUMBERS; i++){
        possibilities *= numbers[i];
    }
    
    uint32_t  divs[QTD_NUMBERS + 1];
    divs[0] = possibilities;
    for(i = 0; i < QTD_NUMBERS; i++){
        divs[i + 1] = divs[i] / numbers[i];
    }
                for(i = 0; i < QTD_NUMBERS + 1; i++){ //DEBUG
                    std::cout << divs[i] << ", ";
                }
                std::cout << std::endl;

    // uint32_t arrSize = possibilities * QTD_NUMBERS * sizeof(uint8_t);
    uint32_t MAXgTid = possibilities * QTD_NUMBERS * sizeof(uint8_t);
    uint32_t arrSize = ((MAX_SAVES + QTD_NUMBERS -1) / QTD_NUMBERS) * QTD_NUMBERS;
                std::cout << "arrSize: " << arrSize << std::endl; //DEBUG
    uint8_t* sequencies = (uint8_t*)malloc(arrSize);

    uint32_t* GPUcont = nullptr;
    uint32_t* GPUdivs = nullptr;
    uint8_t* GPUsequencies = nullptr;    

    NumBlocks = (possibilities + ThrPerBlk - 1) / ThrPerBlk;
                std::cout << "NumBlocks: " << NumBlocks << ", ThrPerBlk: " << ThrPerBlk << std::endl; // debug
                // getchar();

    cudaEventCreate(&time1); 
    cudaEventCreate(&time2);
    cudaEventCreate(&time3);

	cudaEventRecord(time1, 0);
	gpuErrchk(cudaMalloc((void **)&GPUcont, sizeof(uint32_t)));
    gpuErrchk(cudaMemset(GPUcont, 0, sizeof(uint32_t)));

	gpuErrchk(cudaMalloc((void **)&GPUsequencies, arrSize));
    gpuErrchk(cudaMalloc((void **)&GPUdivs, (QTD_NUMBERS + 1) * sizeof(uint32_t)));
    gpuErrchk(cudaMemcpy(GPUdivs, divs, (QTD_NUMBERS + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

	GenSeq <<< NumBlocks, ThrPerBlk >>> (GPUdivs, GPUsequencies, arrSize, GPUcont, MAXgTid);

	gpuErrchk(cudaDeviceSynchronize());
	cudaEventRecord(time2, 0);

	gpuErrchk(cudaMemcpy(sequencies, GPUsequencies, arrSize, cudaMemcpyDeviceToHost));
	cudaEventRecord(time3, 0);

	cudaEventSynchronize(time1); cudaEventSynchronize(time2);
	cudaEventSynchronize(time3);
	cudaEventElapsedTime(&totalTime, time1, time3);
	cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
	cudaEventElapsedTime(&tfrGPUtoCPU, time2, time3);

	gpuErrchk(cudaDeviceSynchronize());
    
    for(i = 0; i < arrSize; i++){
        if(i % QTD_NUMBERS == 0)
            std::cout << std::endl;
        else
            std::cout << ", ";

        std::cout << int(sequencies[i]); 
    }

	printf("\n\n-------------------- ... ----------------------------\n");
	printf("Kernel Execution = %5.2f ms\n",   kernelExecutionTime);
    printf("GPU->CPU Transfer = %5.2f ms ... %4d MB ... %6.2f GB/s\n",  tfrGPUtoCPU, arrSize / 1024 / 1024, 
                                                                        (float)arrSize / (tfrGPUtoCPU * 1024.0 * 1024.0));
	printf("Total time elapsed = %5.2f ms\n", totalTime);
	printf("-------------------- ... ----------------------------\n");


    cudaFree(GPUdivs);
	cudaFree(GPUsequencies);
    cudaEventDestroy(time1); 
    cudaEventDestroy(time2);
	cudaEventDestroy(time3);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess){
		std::cout << "cudaDeviceReset failed!";
		free(sequencies); exit(EXIT_FAILURE);
	}
	free(sequencies);

	return(EXIT_SUCCESS);
}