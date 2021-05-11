
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <algorithm> 
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>   


#define MAX_SAVES 1000
#define QTD_ROOMS 3

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

struct uint8Pt{
	uint8_t *CPU;
	uint8_t *GPU;
};

struct int16Pt{
	int16_t *CPU;
	int16_t *GPU;
};

typedef struct point point;
typedef struct rect rect;
typedef struct uint32Pt uint32Pt;
typedef struct uint8Pt uint8Pt;
typedef struct int16Pt int16Pt;

//---GPU CODE---//
//fact est utilisé pour former la permutation actuelle
//n est la nTh permutation
__global__
void GenSeq(uint32_t *fact, uint8_t *rooms,  int16_t *result, const uint32_t n, const uint32_t offset, const uint32_t maxMYgtid){
	int8_t i, j;
	uint8_t perm[QTD_ROOMS];
	uint32_t ThrPerBlk = blockDim.x;
	uint32_t MYbid = blockIdx.x;
	uint32_t MYtid = threadIdx.x;
    uint32_t MYgtid = ThrPerBlk * MYbid + MYtid + offset; //overflow possible?

	if(MYgtid >= maxMYgtid)
		return;

	//premier partie de calculer la permutation
	//calcule le factoriel code
	i = n;
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

	// uint8_t p1 = 0, p2 = 0, p3 = 0, p4 = 0;
	// uint8_t p_p1 = 0, p_p2 = 0, p_p3 = 0, p_p4 = 0;
	uint8_t curr_room = 0;
	uint32_t curr_result = 0;
	uint8_t p0 = 0;
	uint8_t p1 = 0;
	uint8_t alt = 0, lar = 0, prev_lar = 0, prev_alt = 0;
	int16_t prev_x = 0, prev_y = 0;

	// int16_t temp_result[4*QTD_ROOMS];

	prev_alt = rooms[curr_room++];
	prev_lar = rooms[curr_room++];
	if(MYgtid & 1){
		prev_alt ^= prev_lar;
		prev_lar ^= prev_alt;
		prev_alt ^= prev_lar;
	}

	// curr_result = 0;
	prev_x = 0; prev_y = 0;
	curr_result = n*maxMYgtid + MYgtid*4*QTD_ROOMS;
	result[curr_result++] = prev_x; 
	result[curr_result++] = prev_y;
	result[curr_result++] = prev_alt; 
	result[curr_result++] = prev_lar;

	MYgtid = MYgtid >> 1;
	for(i = 1; i < QTD_ROOMS; i++){
		alt = rooms[curr_room++];
		lar = rooms[curr_room++];

		//si est rotacione, swap altura avec largura
		if(MYgtid & 1){
			alt ^= lar;
			lar ^= alt;
			alt ^= lar;
		}

		p0 = (MYgtid >> 1) & 3;
		p1 = (MYgtid >> 3) & 3;

		prev_x = prev_x + (p0 & 1)*prev_lar - (p1 & 1)*lar;
		prev_y = prev_y + (p0 >> 1)*prev_alt - (p1 >> 1)*lar;

		prev_lar = lar;
		prev_alt = alt;
		result[curr_result++] = prev_x; 
		result[curr_result++] = prev_y;
		result[curr_result++] = alt; 
		result[curr_result++] = lar;
		MYgtid = MYgtid >> 5;
	}
	
	// uint32_t currIndex = 4*QTD_ROOMS*n;
	// for(i = 0; i < 4*QTD_ROOMS; i++){
	// 	result[currIndex++] = temp_result[i];
	// }
	// if(n == 0)
	// curr_result = n*maxMYgtid + debug*4*QTD_ROOMS;
	// // curr_result = 0;
	// printf("%4d - %2d %2d %2d %2d  -  %2d %2d %2d %2d  -  %2d %2d %2d %2d\n", debug, result[curr_result++], result[curr_result++], result[curr_result++], result[curr_result++], result[curr_result++], result[curr_result++], result[curr_result++], result[curr_result++], result[curr_result++], result[curr_result++], result[curr_result++], result[curr_result++]);
	// printf("%d - %d %d %d %d\n", MYgtid, perm[0], perm[1], perm[2], perm[3]);
}

//---CPU CODE---//
int *roomsSeq;

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


__host__
void setupRooms(uint8Pt *rooms){
	uint8_t alturas[] = {60, 30, 15, 10, 10, 10, 10, 10};
	uint8_t larguras[]  = {40, 20, 10, 05, 05, 05, 05, 05};	

	rooms->CPU = (uint8_t*)calloc(QTD_ROOMS*2, sizeof(uint8_t));
	int i = 0;
	for(i = 0; i < QTD_ROOMS; i++){
		rooms->CPU[i*2] = alturas[i];
		rooms->CPU[i*2 + 1] = larguras[i];
	}

	gpuErrchk(cudaMalloc((void **)&(rooms->GPU), QTD_ROOMS * 2 * sizeof(uint32_t)));
	gpuErrchk(cudaMemcpy((rooms->GPU), rooms->CPU, QTD_ROOMS * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

int main(){
	cudaEvent_t *GPUtimes;
    uint32_t ThrPerBlk = 64, NumBlocks = 0;
	int i = 0, j = 0, k = 0;
	int qtdTimes = 3;

	int32_t permutations = factorial(QTD_ROOMS);
	uint32_t roomsAtPerm = pow(2,QTD_ROOMS) * pow(4, (QTD_ROOMS - 1)*2);
	uint32_t possibilities = permutations * roomsAtPerm;
	uint32_t resultSize = possibilities * 4;

	initCuda(&GPUtimes, qtdTimes);

	//compute la quantite de blocks
    NumBlocks = (roomsAtPerm + ThrPerBlk - 1) / ThrPerBlk;
	std::cout << "NumBlocks: " << NumBlocks << ", ThrPerBlk: " << ThrPerBlk << ", Permutations: " << permutations << ", roomsAtPerm: " << roomsAtPerm << ", possibilities: " << possibilities << ", resultSize: " << resultSize << std::endl; // debug

	if(QTD_ROOMS > 6){
		std::cout << "There is not enough ram" << std::endl;
		return EXIT_FAILURE;
	}

	uint8Pt rooms;
	setupRooms(&rooms);

   	//compute factorial numbers
	uint32Pt fact;
	fact.CPU = (uint32_t*)calloc(QTD_ROOMS, sizeof(uint32_t));
	fact.CPU[i] = 1;
	while (i++ < QTD_ROOMS)
	   fact.CPU[i] = fact.CPU[i - 1] * i;

	gpuErrchk(cudaMalloc((void **)&(fact.GPU), QTD_ROOMS * sizeof(uint32_t)));
	gpuErrchk(cudaMemcpy((fact.GPU), fact.CPU, QTD_ROOMS * sizeof(uint32_t), cudaMemcpyHostToDevice));
	
	int16Pt result;
	result.CPU = (int16_t*)calloc(resultSize, sizeof(int16_t));
	gpuErrchk(cudaMalloc((void **)&(result.GPU), resultSize*sizeof(int16_t)));

	
	// void GenSeq(uint32_t *fact, uint8_t *rooms,  int16_t *result, const uint32_t n, const uint32_t offset, const uint32_t maxMYgtid)
	for(i = 0; i < 1; i++){
		GenSeq <<< NumBlocks, ThrPerBlk >>> (fact.GPU, rooms.GPU, result.GPU, i, 0, roomsAtPerm);	
	}
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(result.CPU, result.GPU, resultSize*sizeof(int16_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	std::cout << "\n\nCPU\n\n";
	for(i = 0; i < 1; i++){
		for(j = 0; j <roomsAtPerm; j++){
			std::cout << i*roomsAtPerm + j << "  -\t";
			for(k = 0; k < 4*QTD_ROOMS; k++){
				if(k % 4 == 0)
					std::cout << "\t";
				std::cout << result.CPU[i*roomsAtPerm + j*4*QTD_ROOMS + k] << "  ";
			}
			std::cout << std::endl;
			// std::cout << result.CPU[i*roomsAtPerm + j] << "  ";
			// if(j % 4 == 0)
			// 	std::cout << "    ";
			// if(j % (4*QTD_ROOMS) == 0)
			// 	std::cout << "\n" << (i*roomsAtPerm + j)/(4*QTD_ROOMS) << "  -  ";
		}
	}

	free(fact.CPU);
	free(rooms.CPU);
	free(result.CPU);
	cudaFree(fact.GPU);
	cudaFree(rooms.GPU);
	cudaFree(result.GPU);

	endCuda(&GPUtimes, qtdTimes);
	return(EXIT_SUCCESS);
}