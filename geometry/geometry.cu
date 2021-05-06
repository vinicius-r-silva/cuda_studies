
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <iostream>
#include <algorithm> 
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


#define MAX_SAVES 1000
#define QTD_ROOMS 4

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct point{
    int x;
    int y;
};

struct rect{
    point pt;
    int alt;
    int lar;
};

struct uint32Pt{
	uint32_t *CPU;
	uint32_t *GPU;
};

typedef struct point point;
typedef struct rect rect;
typedef struct uint32Pt uint32Pt;

//---GPU CODE---//
//fact est utilisé pour former la permutation actuelle
//possibilities est la quantité de permutations disponibles
__global__
void GenSeq(uint32_t *fact, const uint32_t possibilities){
	int8_t i, j;
	uint8_t perm[QTD_ROOMS];
	uint32_t ThrPerBlk = blockDim.x;
	uint32_t MYbid = blockIdx.x;
	uint32_t MYtid = threadIdx.x;
    uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;

	if(MYgtid >= possibilities)
		return;

	// compute factorial code
	i = MYgtid;
	for (j = 0; j < QTD_ROOMS; ++j){
		perm[j] = i / fact[QTD_ROOMS - 1 - j];
		i = i % fact[QTD_ROOMS - 1 - j];
	}

	// readjust values to obtain the permutation
	// start from the end and check if preceding values are lower
	for (i = QTD_ROOMS - 1; i > 0; --i)
		for (j = i - 1; j >= 0; --j)
			if (perm[j] <= perm[i])
				perm[i]++;

	// printf("%d - %d %d %d %d\n", MYgtid, perm[0], perm[1], perm[2], perm[3]);
}

// __global__
// void GenSeq(uint32_t *divs, uint8_t*arr, uint32_t arrSize, uint32_t* index, uint32_t MAXgTid){
//     uint32_t ThrPerBlk = blockDim.x;
// 	uint32_t MYbid = blockIdx.x;
// 	uint32_t MYtid = threadIdx.x;
//     uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;
   
//     uint32_t offset = MYgtid;
//     if(offset >= 0)
//         return;
// }   

//---CPU CODE---//
int *roomsSeq;

//le vector roomsSeq représente la actuel séquence de chambres
//initialise le vector pour pouver utiliser la funcion next_permutation à l'avenir
// __host__
// void initPermutation(){
// 	int i = 0;
// 	roomsSeq = (int*)calloc(QTD_ROOMS, sizeof(int));

// 	for(i = 0; i< QTD_ROOMS; i++){
// 		roomsSeq[i] = i;
// 	}
// 	std::sort(roomsSeq, roomsSeq + QTD_ROOMS);
// }

//calcule le valeur de factorial
__host__
int factorial(int x){
	if(x <= 1)
		return 1;
	
	return x*factorial(x-1);
}

//vérifie si au moins une GPU est disponible
//initialise les variables de temps (GPUtimes)
__host__
void initCuda(cudaEvent_t **GPUtimes, int qtdTimes){
	int i = 0;
	int NumGPUs = 0; cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		std::cout << "\nNo CUDA Device is available\n";
		exit(EXIT_FAILURE);
	}

	cudaEvent_t *tempTimes = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * qtdTimes);
	for(i = 0; i < qtdTimes; i++){
		cudaEventCreate(&(tempTimes[i]));
	}
	*GPUtimes = tempTimes;
}

//appelle la funcion cudaEventDestroy pour tout les GPUTimes
//appelle la funcion cudaDeviceReset pour arreter de utilizer la GPU
__host__
void endCuda(cudaEvent_t **GPUtimes, int qtdTimes){
	int i = 0;
	for(i = 0; i < qtdTimes; i++){
		cudaEventDestroy((*GPUtimes)[i]);
	}
	*GPUtimes = nullptr;
	
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess){
		std::cout << "cudaDeviceReset failed!";
		exit(EXIT_FAILURE);
	}
}

int main(){
	cudaEvent_t *GPUtimes;
    uint32_t ThrPerBlk = 32, NumBlocks = 0;
	int i = 0;
	int qtdTimes = 3;
	int possibilities = factorial(QTD_ROOMS);

	initCuda(&GPUtimes, qtdTimes);

   	//compute factorial numbers
	uint32Pt fact;
	fact.CPU = (uint32_t*)calloc(QTD_ROOMS, sizeof(uint32_t));
	fact.CPU[i] = 1;
	while (i++ < QTD_ROOMS)
	   fact.CPU[i] = fact.CPU[i - 1] * i;

	//compute la quantite de blocks
    NumBlocks = (possibilities + ThrPerBlk - 1) / ThrPerBlk;
	std::cout << "NumBlocks: " << NumBlocks << ", ThrPerBlk: " << ThrPerBlk << std::endl; // debug

	gpuErrchk(cudaMalloc((void **)&(fact.GPU), QTD_ROOMS * sizeof(uint32_t)));
	gpuErrchk(cudaMemcpy((fact.GPU), fact.CPU, QTD_ROOMS * sizeof(uint32_t), cudaMemcpyHostToDevice));
	
	GenSeq <<< NumBlocks, ThrPerBlk >>> (fact.GPU, possibilities);
	gpuErrchk(cudaDeviceSynchronize());
	// initPermutation();

	// //-------------------PERMUTATION
	// do{
	// 	for(i = 0; i < QTD_ROOMS; i++){
	// 		std::cout << roomsSeq[i] << "  ";
	// 	}
	// 	std::cout << std::endl;
	// }while(std::next_permutation(roomsSeq, roomsSeq + QTD_ROOMS));

	

    // uint32_t* GPUcont = nullptr;
    // uint32_t* GPUdivs = nullptr;
    // uint8_t* GPUsequencies = nullptr;    
	// cudaEventRecord(time1, 0);
   
	// gpuErrchk(cudaMalloc((void **)&GPUcont, sizeof(uint32_t)));
    // gpuErrchk(cudaMemset(GPUcont, 0, sizeof(uint32_t)));

	// gpuErrchk(cudaMalloc((void **)&GPUsequencies, arrSize));
    // gpuErrchk(cudaMalloc((void **)&GPUdivs, (QTD_NUMBERS + 1) * sizeof(uint32_t)));
    // gpuErrchk(cudaMemcpy(GPUdivs, divs, (QTD_NUMBERS + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

	// GenSeq <<< NumBlocks, ThrPerBlk >>> (GPUdivs, GPUsequencies, arrSize, GPUcont, MAXgTid);

	// gpuErrchk(cudaDeviceSynchronize());
	// cudaEventRecord(time2, 0);

	// gpuErrchk(cudaMemcpy(sequencies, GPUsequencies, arrSize, cudaMemcpyDeviceToHost));
	// cudaEventRecord(time3, 0);

	// float totalTime, kernelExecutionTime, tfrGPUtoCPU;
	// cudaEventSynchronize(time1); cudaEventSynchronize(time2);
	// cudaEventSynchronize(time3);
	// cudaEventElapsedTime(&totalTime, time1, time3);
	// cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
	// cudaEventElapsedTime(&tfrGPUtoCPU, time2, time3);

	// gpuErrchk(cudaDeviceSynchronize());

	// printf("\n\n-------------------- ... ----------------------------\n");
	// printf("Kernel Execution = %5.2f ms\n",   kernelExecutionTime);
    // printf("GPU->CPU Transfer = %5.2f ms ... %4d MB ... %6.2f GB/s\n",  tfrGPUtoCPU, arrSize / 1024 / 1024, 
    //                                                                     (float)arrSize / (tfrGPUtoCPU * 1024.0 * 1024.0));
	// printf("Total time elapsed = %5.2f ms\n", totalTime);
	// printf("-------------------- ... ----------------------------\n");

    // cudaFree(GPUdivs);
	// cudaFree(GPUsequencies);

	free(fact.CPU);
	cudaFree(fact.GPU);

	endCuda(&GPUtimes, qtdTimes);
	return(EXIT_SUCCESS);
}