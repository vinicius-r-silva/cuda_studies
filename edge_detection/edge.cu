
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

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

int main(){
	cudaError_t cudaStatus, cudaStatus2;
	cudaEvent_t time1, time2, time3, time4;
	cudaDeviceProp GPUprop;

	uint32_t SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;
	char SupportedBlocks[100];

	uint32_t BlkPerRow, ThrPerBlk = 128, NumBlocks, GPUDataTransfer;
	float totalTime, tfrCPUtoGPU, kernelExecutionTime, tfrGPUtoCPU;

	char InputFileName[] = "../img/img.bmp";
    char OutputFileName[] = "../img/flip.bmp";


    exit(EXIT_SUCESS);
}