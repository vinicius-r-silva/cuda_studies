
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

// #include "cudaDefines.h"

struct ImgProp {
	uint32_t Hpixels;
	uint32_t Vpixels;

	uint8_t HeaderInfo[14];
	uint8_t* HeaderMeta;
	uint16_t HeaderMetaSize;

	uint32_t Hbytes;
	uint32_t IMAGESIZE;
};

struct ImgProp ip;
uint8_t* TheImg, * CpyImg;
uint8_t* GPUImg, * GPUCopyImg;

uint8_t* ReadBMPlin(char* fn) {
	static uint8_t* Img;
    FILE* f = fopen(fn, "rb");
	if (f == NULL) { printf("\n\n%s NOT FOUND\n\n", fn); exit(EXIT_FAILURE); }
	uint8_t HeaderInfo[14];
	fread(HeaderInfo, sizeof(uint8_t), 14, f); // read the 54-byte header
	ip.HeaderMetaSize = *(int*)&HeaderInfo[10];
	ip.HeaderMeta = (uint8_t*)malloc(ip.HeaderMetaSize * sizeof(uint8_t));
	fread(ip.HeaderMeta, sizeof(uint8_t), ip.HeaderMetaSize, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&(ip.HeaderMeta[4]); ip.Hpixels = width;
	int height = *(int*)&(ip.HeaderMeta[8]); ip.Vpixels = height;
	//int RowBytes = (width * 3 + 3) & (~3); ip.Hbytes = RowBytes;
	int RowBytes = ip.Hpixels * 3; ip.Hbytes = RowBytes;

	ip.IMAGESIZE = ip.Hbytes * ip.Vpixels;
	memcpy(ip.HeaderInfo, HeaderInfo, 14); //save header for re-use
	printf("\n Input File name: %17s\n\nHeaderMetaSize: %u, Hb: %u, Hp: %u, Vp: %u, File Size=%u\n\n", fn,
		ip.HeaderMetaSize, ip.Hbytes, ip.Hpixels, ip.Vpixels, ip.IMAGESIZE);
	// allocate memory to store the main image (1 Dimensional array)
	Img = (uint8_t*)malloc(ip.IMAGESIZE);
	if (Img == NULL) return Img; // Cannot allocate memory
	// read the image from disk
	fread(Img, sizeof(uint8_t), ip.IMAGESIZE, f); fclose(f); return Img;
}

// Write the 1D linear-memory stored image into file.
void WriteBMPlin(uint8_t* Img, char* fn) {
	FILE* f = fopen(fn, "wb");
	if (f == NULL) { printf("\n\nFILE CREATION ERROR: %s\n\n", fn); exit(1); }
	fwrite(ip.HeaderInfo, sizeof(uint8_t), 14, f); //write header
	fwrite(ip.HeaderMeta, sizeof(uint8_t), ip.HeaderMetaSize, f); //write header
	fwrite(Img, sizeof(uint8_t), ip.IMAGESIZE, f); //write data
	printf("\nOutput File name: %17s (%u x %u) File Size=%u\n\n", fn, ip.Hpixels,
		ip.Vpixels, ip.IMAGESIZE);
	fclose(f);
}

__global__
void Vflip(uint8_t* ImgDst, uint8_t* ImgSrc, uint32_t Hpixels, uint32_t Vpixels) {
	uint32_t ThrPerBlk = blockDim.x;
	uint32_t MYbid = blockIdx.x;
	uint32_t MYtid = threadIdx.x;
	uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;
	uint32_t BlkPerRow = (Hpixels + ThrPerBlk - 1) / ThrPerBlk; // ceil
	uint32_t RowBytes = (Hpixels * 3 + 3) & (~3);
	uint32_t MYrow = MYbid / BlkPerRow;
	uint32_t MYcol = MYgtid - MYrow * BlkPerRow * ThrPerBlk;
	if (MYcol >= Hpixels) return; // col out of range
	uint32_t MYmirrorrow = Vpixels - 1 - MYrow;
	uint32_t MYsrcOffset = MYrow * RowBytes;
	uint32_t MYdstOffset = MYmirrorrow * RowBytes;
	uint32_t MYsrcIndex = MYsrcOffset + 3 * MYcol;
	uint32_t MYdstIndex = MYdstOffset + 3 * MYcol;
	// swap pixels RGB @MYcol , @MYmirrorcol
	ImgDst[MYdstIndex] = ImgSrc[MYsrcIndex];
	ImgDst[MYdstIndex + 1] = ImgSrc[MYsrcIndex + 1];
	ImgDst[MYdstIndex + 2] = ImgSrc[MYsrcIndex + 2];
}

int main() {
	cudaError_t cudaStatus, cudaStatus2;
	cudaEvent_t time1, time2, time3, time4;
	cudaDeviceProp GPUprop;

	uint32_t SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;
	char SupportedBlocks[100];

	uint32_t BlkPerRow, ThrPerBlk = 128, NumBlocks, GPUDataTransfer;
	float totalTime, tfrCPUtoGPU, kernelExecutionTime, tfrGPUtoCPU;

	char InputFileName[] = "../img/img.bmp";
    char OutputFileName[] = "../img/flip.bmp";

	TheImg = ReadBMPlin(InputFileName);
	CpyImg = (uint8_t*)malloc(ip.IMAGESIZE);
	//WriteBMPlin(TheImg, OutputFileName);
	//return 0;

	int NumGPUs = 0; cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0) {
		std::cout << "\nNo CUDA Device is available\n";
		exit(EXIT_FAILURE);
	}

	cudaGetDeviceProperties(&GPUprop, 0);
	SupportedKBlocks = ((uint32_t)GPUprop.maxGridSize[0] * (uint32_t)GPUprop.maxGridSize[1] *
		(uint32_t)GPUprop.maxGridSize[2]) / 1024;

	SupportedMBlocks = SupportedKBlocks / 1024;
	sprintf(SupportedBlocks, "%u %c", (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks,
		(SupportedMBlocks >= 5) ? 'M' : 'K');
	MaxThrPerBlk = (uint32_t)GPUprop.maxThreadsPerBlock;

	cudaEventCreate(&time1); cudaEventCreate(&time2);
	cudaEventCreate(&time3); cudaEventCreate(&time4);

	BlkPerRow = (ip.Hpixels + ThrPerBlk - 1) / ThrPerBlk;
	NumBlocks = ip.Vpixels * BlkPerRow;
	GPUDataTransfer = 2 * ip.IMAGESIZE;

	cudaEventRecord(time1, 0);
	cudaStatus = cudaMalloc((void**)&GPUImg, ip.IMAGESIZE);
	cudaStatus2 = cudaMalloc((void**)&GPUCopyImg, ip.IMAGESIZE);
	if ((cudaStatus != cudaSuccess) || (cudaStatus2 != cudaSuccess)) {
		std::cout << "cudaMalloc failed! Can't allocate GPU memory";
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpy(GPUImg, TheImg, ip.IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMemCpy CPU to GPU failed!";
		exit(EXIT_FAILURE);
	}
	cudaEventRecord(time2, 0);

	Vflip <<<NumBlocks, ThrPerBlk>>> (GPUCopyImg, GPUImg, ip.Hpixels, ip.Vpixels);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceSynchronize error code " << cudaStatus << " ...\n";
		exit(EXIT_FAILURE);
	}
	cudaEventRecord(time3, 0);

	cudaStatus = cudaMemcpy(CpyImg, GPUCopyImg, ip.IMAGESIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMemCpy GPU to CPU failed!" << cudaStatus;
		exit(EXIT_FAILURE);
	}
	cudaEventRecord(time4, 0);

	cudaEventSynchronize(time1); cudaEventSynchronize(time2);
	cudaEventSynchronize(time3); cudaEventSynchronize(time4);
	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
	cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Program failed afeter cudaDeviceSyncronize()";
		free(TheImg); free(CpyImg);
		exit(EXIT_FAILURE);
	}

	WriteBMPlin(CpyImg, OutputFileName);

	printf("--...--\n"); printf("%s ComputeCapab=%d.%d [supports max %s blocks]\n",
		GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks); printf("...\n");
	printf("maxTrPerBlk: %d\n", MaxThrPerBlk);
	printf("%s\n %s\n\n ThrPerBlock: %u, Blocks: %u, BlkPerRow: %u\n", InputFileName,
		OutputFileName, ThrPerBlk, NumBlocks, BlkPerRow);
	printf("-------------------- ... ----------------------------\n");
	printf("CPU->GPU Transfer = %5.2f ms ... %4d MB ... %6.2f GB/s\n",
		tfrCPUtoGPU, ip.IMAGESIZE / 1024 / 1024, (float)ip.IMAGESIZE / (tfrCPUtoGPU *
			1024.0 * 1024.0));
	printf("Kernel Execution = %5.2f ms ... %4d MB ... %6.2f GB/s\n",
		kernelExecutionTime, GPUDataTransfer / 1024 / 1024, (float)GPUDataTransfer /
		(kernelExecutionTime * 1024.0 * 1024.0));
	printf("GPU->CPU Transfer = %5.2f ms ... %4d MB ... %6.2f GB/s\n",
		tfrGPUtoCPU, ip.IMAGESIZE / 1024 / 1024, (float)ip.IMAGESIZE / (tfrGPUtoCPU *
			1024.0 * 1024.0));
	printf("Total time elapsed = %5.2f ms\n", totalTime);
	printf("-------------------- ... ----------------------------\n");


	cudaFree(GPUImg); cudaFree(GPUCopyImg);
	cudaEventDestroy(time1); cudaEventDestroy(time2);
	cudaEventDestroy(time3); cudaEventDestroy(time4);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceReset failed!";
		free(TheImg); free(CpyImg); exit(EXIT_FAILURE);
	}
	free(TheImg); free(CpyImg);

	getchar();
	//getchar();
	return(EXIT_SUCCESS);
}