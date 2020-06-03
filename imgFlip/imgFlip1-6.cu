
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

// #include "cudaDefines.h"

#define REPETITON 1000

struct ImgProp {
	uint32_t Hpixels;
	uint32_t Vpixels;

	uint8_t HeaderInfo[14];
	uint8_t* HeaderMeta;
	uint16_t HeaderMetaSize;

	uint32_t Hbytes;
	uint32_t IMAGESIZE;
	uint32_t IMAGEPIX;
};

struct ImgProp ip;
uint8_t* TheImg, * CpyImg;
uint8_t* GPUImg, * GPUCopyImg;

cv::Mat Image;

//-----------------------------------------------------------------------------------------------------------------//
//-----------------------------------------------IMG READ WRITE----------------------------------------------------//
uint8_t* ReadBMPlin(char* fn) {
    Image = cv::imread(fn);

    if(!Image.data){                              // Check for invalid input
        std::cout <<  "Could not open or find the image" << std::endl ;
        exit(EXIT_FAILURE);
    }

    // cv::resize(Image, Image, cv::Size(0,0), 0.5, 0.5);
    ip.Hpixels = Image.cols;
    ip.Vpixels = Image.rows;
    ip.Hbytes = Image.cols * 3;
    
	ip.IMAGESIZE = ip.Hbytes * ip.Vpixels;
    ip.IMAGEPIX = ip.Hpixels * ip.Vpixels;
    
    return Image.data;
}

// Write the 1D linear-memory stored image into file.
void WriteBMPlin(uint8_t* Img, char* fn) {
    memcpy(Image.data, Img, ip.IMAGESIZE);
    imwrite(fn, Image);
}




//-----------------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------VFLIP-------------------------------------------------------//
__global__
void Vflip1(uint8_t* ImgDst, uint8_t* ImgSrc, uint32_t Hpixels, uint32_t Vpixels) {
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

__global__
void Vflip2(uint8_t* ImgDst, uint8_t* ImgSrc, uint32_t Hpixels, uint32_t Vpixels, uint32_t BlkPerRow, uint32_t RowBytes) {
	uint32_t ThrPerBlk = blockDim.x;
	uint32_t MYbid = blockIdx.x;
	uint32_t MYtid = threadIdx.x;
	uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;

	// uint32_t BlkPerRow = (Hpixels + ThrPerBlk - 1) / ThrPerBlk; // ceil
	// uint32_t RowBytes = (Hpixels * 3 + 3) & (~3);

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

__global__
void Vflip3(uint8_t* ImgDst, uint8_t* ImgSrc, uint32_t Hpixels, uint32_t Vpixels, uint32_t RowBytes) {
	uint32_t ThrPerBlk = blockDim.x;
	uint32_t MYbid = blockIdx.x;
	uint32_t MYtid = threadIdx.x;
	// uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;

	// uint32_t BlkPerRow = (Hpixels + ThrPerBlk - 1) / ThrPerBlk; // ceil
	// uint32_t RowBytes = (Hpixels * 3 + 3) & (~3);

	// uint32_t MYrow = MYbid / BlkPerRow;
	// uint32_t MYcol = MYgtid - MYrow * BlkPerRow * ThrPerBlk;
	uint32_t MYrow = blockIdx.y;
	uint32_t MYcol = MYbid * ThrPerBlk + MYtid;
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

__global__
void Vflip4(uint8_t* ImgDst, uint8_t* ImgSrc, uint32_t Hpixels, uint32_t Vpixels, uint32_t RowBytes) {
	uint32_t ThrPerBlk = blockDim.x;
	uint32_t MYbid = blockIdx.x;
	uint32_t MYtid = threadIdx.x;
	// uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;

	// uint32_t BlkPerRow = (Hpixels + ThrPerBlk - 1) / ThrPerBlk; // ceil
	// uint32_t RowBytes = (Hpixels * 3 + 3) & (~3);

	// uint32_t MYrow = MYbid / BlkPerRow;
	// uint32_t MYcol = MYgtid - MYrow * BlkPerRow * ThrPerBlk;
	uint32_t MYrow = blockIdx.y;
	uint32_t MYcol = (MYbid * ThrPerBlk + MYtid) * 2;
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

	if(MYcol + 1 > Hpixels) return;
	ImgDst[MYdstIndex + 3] = ImgSrc[MYsrcIndex + 3];
	ImgDst[MYdstIndex + 4] = ImgSrc[MYsrcIndex + 4];
	ImgDst[MYdstIndex + 5] = ImgSrc[MYsrcIndex + 5];
}

__global__
void Vflip5(uint8_t* ImgDst, uint8_t* ImgSrc, uint32_t Hpixels, uint32_t Vpixels, uint32_t RowBytes) {
	uint32_t ThrPerBlk = blockDim.x;
	uint32_t MYbid = blockIdx.x;
	uint32_t MYtid = threadIdx.x;
	// uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;

	// uint32_t BlkPerRow = (Hpixels + ThrPerBlk - 1) / ThrPerBlk; // ceil
	// uint32_t RowBytes = (Hpixels * 3 + 3) & (~3);

	// uint32_t MYrow = MYbid / BlkPerRow;
	// uint32_t MYcol = MYgtid - MYrow * BlkPerRow * ThrPerBlk;
	uint32_t MYrow = blockIdx.y;
	uint32_t MYcol = (MYbid * ThrPerBlk + MYtid) * 4;
	if (MYcol >= Hpixels) return; // col out of range

	uint32_t MYmirrorrow = Vpixels - 1 - MYrow;
	uint32_t MYsrcOffset = MYrow * RowBytes;
	uint32_t MYdstOffset = MYmirrorrow * RowBytes;
	uint32_t MYsrcIndex = MYsrcOffset + 3 * MYcol;
	uint32_t MYdstIndex = MYdstOffset + 3 * MYcol;

	for(uint32_t a = 0; a < 4; a++){
		ImgDst[MYdstIndex] = ImgSrc[MYsrcIndex];
		ImgDst[MYdstIndex + 1] = ImgSrc[MYsrcIndex + 1];
		ImgDst[MYdstIndex + 2] = ImgSrc[MYsrcIndex + 2];
		MYdstIndex += 3;
		MYsrcIndex += 3;
		MYcol++;
		if(MYcol > Hpixels) return;
	}
}

__global__
void Vflip6(uint8_t* ImgDst, uint8_t* ImgSrc, uint32_t Hpixels, uint32_t Vpixels, uint32_t RowBytes) {
	__shared__ uint8_t PixBuffer[3072];

	uint32_t ThrPerBlk = blockDim.x;
	uint32_t MYbid = blockIdx.x;
	uint32_t MYtid = threadIdx.x;
	// uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;

	// uint32_t BlkPerRow = (Hpixels + ThrPerBlk - 1) / ThrPerBlk; // ceil
	// uint32_t RowBytes = (Hpixels * 3 + 3) & (~3);

	// uint32_t MYrow = MYbid / BlkPerRow;
	// uint32_t MYcol = MYgtid - MYrow * BlkPerRow * ThrPerBlk;
	uint32_t MYrow = blockIdx.y;
	uint32_t MYcol = MYbid * ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return; // col out of range

	MYtid *= 3;
	uint32_t MYmirrorrow = Vpixels - 1 - MYrow;
	uint32_t MYsrcOffset = MYrow * RowBytes;
	uint32_t MYdstOffset = MYmirrorrow * RowBytes;
	uint32_t MYsrcIndex = MYsrcOffset + 3 * MYcol;
	uint32_t MYdstIndex = MYdstOffset + 3 * MYcol;

	PixBuffer[MYtid] = ImgSrc[MYsrcIndex];
	PixBuffer[MYtid + 1] = ImgSrc[MYsrcIndex + 1];
	PixBuffer[MYtid + 2] = ImgSrc[MYsrcIndex + 2];
	__syncthreads();
	ImgDst[MYdstIndex] = PixBuffer[MYtid];
	ImgDst[MYdstIndex + 1] = PixBuffer[MYtid + 1];
	ImgDst[MYdstIndex + 2] = PixBuffer[MYtid + 2];
	
}



//-----------------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------MAIN--------------------------------------------------------//
int main(int argc, char *argv[]){
	cudaError_t cudaStatus, cudaStatus2;
	cudaEvent_t time1, time2, time3, time4;
	cudaDeviceProp GPUprop;

	int KernelNumber = 1;
	if(argc > 1)
		KernelNumber = atoi(argv[1]);

	uint32_t SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;
	char SupportedBlocks[100];

	uint32_t BlkPerRow, ThrPerBlk = 128, NumBlocks, GPUDataTransfer;
	float totalTime, tfrCPUtoGPU, kernelExecutionTime, tfrGPUtoCPU;

	char InputFileName[] = "../img/img.bmp";
	char OutputFileName[] = "../img/flip.bmp";
	char KernelName[100]; memset(KernelName, '\0', 100);

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


	// uint32_t RowBytes = (ip.Hpixels * 3 + 3) & (~3);
	uint32_t RowBytes = ip.Hbytes;
	int i = 0;
	cudaEventRecord(time2, 0);

	if(KernelNumber == 1){
		for(i = 0; i < REPETITON; i++){
			Vflip1 <<<NumBlocks, ThrPerBlk>>> (GPUCopyImg, GPUImg, ip.Hpixels, ip.Vpixels);
		}
		strcpy(KernelName,"VFlip1: Each thread copies 1 pixel (using a 1D grid)");
	}
	else if(KernelNumber == 2){
		for(i = 0; i < REPETITON; i++){
			Vflip2 <<<NumBlocks, ThrPerBlk>>> (GPUCopyImg, GPUImg, ip.Hpixels, ip.Vpixels, BlkPerRow, RowBytes);
		}
		strcpy(KernelName,"VFlip2: Each thread copies 1 pixel (using a 1D grid)");
	}
	else if(KernelNumber == 3){
		dim3 dimGrid2D(BlkPerRow, ip.Vpixels);
		for(i = 0; i < REPETITON; i++){
			Vflip3 <<<dimGrid2D, ThrPerBlk>>> (GPUCopyImg, GPUImg, ip.Hpixels, ip.Vpixels, RowBytes);
		}
		strcpy(KernelName,"VFlip3: Each thread copies 1 pixel (using a 2D grid)");
	}
	else if(KernelNumber == 4){
		BlkPerRow = (BlkPerRow + 2 -1) / 2;
		dim3 dimGrid2D(BlkPerRow, ip.Vpixels);
		for(i = 0; i < REPETITON; i++){
			Vflip4 <<<dimGrid2D, ThrPerBlk>>> (GPUCopyImg, GPUImg, ip.Hpixels, ip.Vpixels, RowBytes);
		}
		strcpy(KernelName,"VFlip4: Each thread copies 1 pixel (using a 2D grid)");
	}
	else if(KernelNumber == 5){
		BlkPerRow = (BlkPerRow + 4 -1) / 4;
		dim3 dimGrid2D(BlkPerRow, ip.Vpixels);
		for(i = 0; i < REPETITON; i++){
			Vflip5 <<<dimGrid2D, ThrPerBlk>>> (GPUCopyImg, GPUImg, ip.Hpixels, ip.Vpixels, RowBytes);
		}
		strcpy(KernelName,"VFlip4: Each thread copies 1 pixel (using a 2D grid)");
	}
	else if(KernelNumber == 6){
		dim3 dimGrid2D(BlkPerRow, ip.Vpixels);
		for(i = 0; i < REPETITON; i++){
			Vflip6 <<<dimGrid2D, ThrPerBlk>>> (GPUCopyImg, GPUImg, ip.Hpixels, ip.Vpixels, RowBytes);
		}
		strcpy(KernelName,"VFlip6: Each thread copies 1 pixel (using a 2D grid)");
	}
	else{
		printf("Unkown Kernel Number: %d\n", KernelNumber);

		cudaFree(GPUImg); cudaFree(GPUCopyImg);
		cudaEventDestroy(time1); cudaEventDestroy(time2);
		cudaEventDestroy(time3); cudaEventDestroy(time4);
	
		cudaStatus = cudaDeviceReset();
		free(CpyImg);
		return 1;		
	}


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

	kernelExecutionTime /= REPETITON;

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Program failed afeter cudaDeviceSyncronize()";
		free(TheImg); free(CpyImg);
		exit(EXIT_FAILURE);
	}

	WriteBMPlin(CpyImg, OutputFileName);

	printf("--...--\n"); 
	
	printf("%s\n", KernelName);

	// printf("%s ComputeCapab=%d.%d [supports max %s blocks]\n",
	// 	GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks); printf("...\n");
	printf("maxTrPerBlk: %d\n", MaxThrPerBlk);
	printf("%s %u x %u\n%s\n\nThrPerBlock: %u, Blocks: %u, BlkPerRow: %u\n", InputFileName, ip.Hpixels, ip.Vpixels,
		OutputFileName, ThrPerBlk, NumBlocks, BlkPerRow);
	printf("-------------------- ... ----------------------------\n");
	printf("CPU->GPU Transfer = %5.2f ms ... %4d MB ... %6.2f GB/s\n",
		tfrCPUtoGPU, ip.IMAGESIZE / 1024 / 1024, (float)ip.IMAGESIZE / (tfrCPUtoGPU *
			1024.0 * 1024.0));
	printf("Kernel Execution = %5.2f ms ... %4d MB ... %6.2f GB/s (%3.2f%%)\n",
		kernelExecutionTime, GPUDataTransfer / 1024 / 1024, (float)GPUDataTransfer /
		(kernelExecutionTime * 1024.0 * 1024.0), float((float)GPUDataTransfer / (kernelExecutionTime * 1024.0 * 1024.0)) / 1.1210);
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
	//free(TheImg); 
	free(CpyImg);

	// getchar();
	//getchar();
	return(EXIT_SUCCESS);
}