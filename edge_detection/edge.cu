
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define CEIL(a,b)  ((a+b-1)/b)
#define IPH        ip.Hpixels
#define IPV        ip.Vpixels

#define EDGE 255
#define NOEDGE 0

#define MB(bytes)     (bytes/1024/1024)
#define BW(bytes,timems) ((float)bytes/(timems * 1.024*1024.0*1024.0))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

int ThreshLo=50, ThreshHi=100;

struct ImgProp ip;
uint8_t* TheImg, * CpyImg;
uint8_t* GPUImg, * GPUResultImg;
double*  GPUptr, *GPUBWImg, *GPUGaussImg, *GPUGradient, *GPUTheta;

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
	ip.IMAGEPIX = ip.Hpixels * ip.Vpixels;
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
void BWKernel(double *ImgBW, uint8_t *ImgGPU, uint32_t Hpixels){
    uint32_t ThrPerBlk = blockDim.x;
    uint32_t MYbid = blockIdx.x;
    uint32_t MYtid = threadIdx.x;
    uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;
    double R, G, B;

    uint32_t BlkPerRow = CEIL(Hpixels, ThrPerBlk);
    uint32_t RowBytes = Hpixels * 3;
    uint32_t MYrow = MYbid / BlkPerRow;
    uint32_t MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
    if (MYcol >= Hpixels)return; // col out of range
    
    uint32_t MYsrcIndex = MYrow * RowBytes + 3 * MYcol;
    uint32_t MYpixIndex = MYrow * Hpixels + MYcol;

    B = (double)ImgGPU[MYsrcIndex];
    G = (double)ImgGPU[MYsrcIndex + 1];
    R = (double)ImgGPU[MYsrcIndex + 2];
    ImgBW[MYpixIndex] = (R+G+B)/3.0;
}

__device__
double Gauss[5][5]={{ 2,   4,    5,  4,   2 },
                    { 4,  9,   12,  9,   4 },
                    { 5,  12,  15,  12,  5 },
                    { 4,  9,   12,  9,   4 },
                    { 2,  4,   5,   4,   2 } };
// Kernel that calculates a Gauss image from the B&W image
// resulting image has a double type for each pixel position

__global__
void GaussKernel(double *ImgGauss, double*ImgBW, uint32_t Hpixels, uint32_t Vpixels){
    uint32_t ThrPerBlk = blockDim.x;uint32_t MYbid = blockIdx.x;
    uint32_t MYtid = threadIdx.x;
    uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;
    int row, col, indx, i, j;
    double G=0.00;
    
    //uint32_t NumBlocks = gridDim.x;
    uint32_t BlkPerRow = CEIL(Hpixels, ThrPerBlk);
    int MYrow = MYbid / BlkPerRow;
    int MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
    if (MYcol >= Hpixels) return; // col out of range
    
    uint32_t MYpixIndex = MYrow * Hpixels + MYcol;
    
    if ((MYrow<2) || (MYrow>Vpixels - 3) || (MYcol<2) || (MYcol>Hpixels - 3)){
        ImgGauss[MYpixIndex] = 0.0;
        return;
    }else{
        G = 0.0;
        for(i = -2; i <= 2; i++){
            for(j = -2; j <= 2; j++){
                row = MYrow + i;
                col = MYcol + j;
                indx = row*Hpixels + col;
                G += (ImgBW[indx] * Gauss[i + 2][j + 2]);
            }
        }
        ImgGauss[MYpixIndex] = G / 159.00;
    }
}



__device__
double Gx[3][3] = { { -1, 0,  1 },
                    { -2, 0,  2 },
                    { -1, 0,  1 } };

__device__
double Gy[3][3] = { { -1, -2, -1 },
                    {  0,  0,  0 },
                    {  1,  2,  1 } };
// Kernel that calculates Gradient, Theta from the Gauss image
// resulting image has a double type for each pixel position

__global__
void SobelKernel(double *ImgGrad, double*ImgTheta,double*ImgGauss, uint32_t Hpixels,uint32_t Vpixels){
    uint32_t ThrPerBlk = blockDim.x;
    uint32_t MYbid = blockIdx.x;
    uint32_t MYtid = threadIdx.x;
    uint32_t MYgtid = ThrPerBlk * MYbid + MYtid;
    int row, col, indx, i, j;
    double GX,GY;

    //uint32_t NumBlocks = gridDim.x;
    uint32_t BlkPerRow = CEIL(Hpixels, ThrPerBlk);
    int MYrow = MYbid / BlkPerRow;
    int MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
    if (MYcol >= Hpixels)return; // col out of range
    
    uint32_t MYpixIndex = MYrow * Hpixels + MYcol;
    if ((MYrow<1) || (MYrow>Vpixels - 2) || (MYcol<1) || (MYcol>Hpixels - 2)){
        ImgGrad[MYpixIndex] = 0.0;
        ImgTheta[MYpixIndex] = 0.0;
        return;
    }
    else{
        GX = 0.0;
        GY = 0.0;
        for(i = -1; i <= 1; i++){
            for(j = -1; j <= 1; j++){
                row = MYrow + i;
                col = MYcol + j;
                indx = row*Hpixels + col;
                GX += (ImgGauss[indx] * Gx[i + 1][j + 1]);
                GY += (ImgGauss[indx] * Gy[i + 1][j + 1]);
            }
        }
        ImgGrad[MYpixIndex] = sqrt(GX*GX + GY*GY);
        ImgTheta[MYpixIndex] = atan(GX / GY)*180.0 / M_PI;
    }
}

// Kernel that calculates the threshold image from Gradient, Theta
// resulting image has an RGB for each pixel, same RGB for each pixel
__global__
void ThresholdKernel(uint8_t *ImgResult, double* ImgGrad, double* ImgTheta, uint8_t Hpixels, uint32_t Vpixels, uint32_t ThreshLo, uint32_t ThreshHi){
    uint32_t ThrPerBlk= blockDim.x;
    uint32_t MYbid= blockIdx.x;
    uint32_t MYtid= threadIdx.x;
    uint32_t MYgtid= ThrPerBlk*MYbid+MYtid;
    
    double L,H,G,T;
    uint8_t PIXVAL;
    uint32_t BlkPerRow= CEIL(Hpixels,ThrPerBlk);
    
    int MYrow = MYbid / BlkPerRow;
    uint32_t RowBytes= Hpixels*3;
    
    int MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
    if (MYcol >= Hpixels)
        return; // col out of range
    
    uint32_t MYresultIndex= MYrow*RowBytes+3*MYcol;
    uint32_t MYpixIndex= MYrow*Hpixels+MYcol;
    if ((MYrow<1) || (MYrow>Vpixels-2) || (MYcol<1) || (MYcol>Hpixels-2)){
        ImgResult[MYresultIndex]= NOEDGE;
        ImgResult[MYresultIndex+1]= NOEDGE;
        ImgResult[MYresultIndex+2]= NOEDGE;
        return;
    }else{
        L = (double)ThreshLo;
        H = (double)ThreshHi;
        G = ImgGrad[MYpixIndex];
        PIXVAL= NOEDGE;
        
        if (G <= L){
            PIXVAL= NOEDGE; // no edge
        
        }else if(G >= H){
            PIXVAL = EDGE; // edge
        
        }else{
            T = ImgTheta[MYpixIndex];
            if ((T<-67.5) || (T>67.5)){ 
                // Look at left and right: [row][col-1] and [row][col+1]
                PIXVAL= ((ImgGrad[MYpixIndex-1]>H) || (ImgGrad[MYpixIndex+1]>H)) ? EDGE :NOEDGE;
            
            }else if((T >= -22.5) && (T <= 22.5)){
                // Look at top and bottom: [row-1][col] and [row+1][col]
                PIXVAL= ((ImgGrad[MYpixIndex-Hpixels]>H) ||(ImgGrad[MYpixIndex+Hpixels]>H)) ? EDGE : NOEDGE;
            
            }else if((T>22.5) && (T <= 67.5)){
                // Look at upper right, lower left: [row-1][col+1] and [row+1][col-1]
                PIXVAL= ((ImgGrad[MYpixIndex-Hpixels+1]>H) ||(ImgGrad[MYpixIndex+Hpixels-1]>H)) ? EDGE : NOEDGE;
            
            }else if((T >= -67.5) && (T<-22.5)){
                // Look at upper left, lower right: [row-1][col-1] and [row+1][col+1]
                PIXVAL=((ImgGrad[MYpixIndex-Hpixels-1]>H) ||(ImgGrad[MYpixIndex+Hpixels+1]>H)) ? EDGE : NOEDGE;
            }
        }
        ImgResult[MYresultIndex]=PIXVAL;
        ImgResult[MYresultIndex+1]=PIXVAL;
        ImgResult[MYresultIndex+2]=PIXVAL;
    }
}

int main(){ 
    cudaEvent_t time1, time2, time2BW, time2Gauss, time2Sobel, time3, time4;
    cudaDeviceProp GPUprop;
    
    int NumGPUs = 0; cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		std::cout << "\nNo CUDA Device is available\n";
		exit(EXIT_FAILURE);
    }
    
	uint32_t SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;
	char SupportedBlocks[100];

    uint32_t BlkPerRow, ThrPerBlk = 256, NumBlocks;
    uint64_t GPUDataTfrBW, GPUDataTfrGauss, GPUDataTfrSobel, GPUDataTfrThresh;
    float totalKernelTime, totalTime, tfrCPUtoGPU, tfrGPUtoCPU;
    float kernelExecTimeBW, kernelExecTimeGauss, kernelExecTimeSobel, kernelExecTimeThreshold;
    float GPUDataTfrKernel, GPUDataTfrTotal;

	char InputFileName[] = "../img/img.bmp";
    char OutputFileName[] = "../img/edge.bmp";

    TheImg = ReadBMPlin(InputFileName);
	CpyImg = (uint8_t*)malloc(ip.IMAGESIZE);

    cudaGetDeviceProperties(&GPUprop, 0);
	SupportedKBlocks = ((uint32_t)GPUprop.maxGridSize[0] * (uint32_t)GPUprop.maxGridSize[1] *
		(uint32_t)GPUprop.maxGridSize[2]) / 1024;

	SupportedMBlocks = SupportedKBlocks / 1024;
	sprintf(SupportedBlocks, "%u %c", (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks,
		(SupportedMBlocks >= 5) ? 'M' : 'K');
    MaxThrPerBlk = (uint32_t)GPUprop.maxThreadsPerBlock;
    
	cudaEventCreate(&time1); cudaEventCreate(&time2);
	cudaEventCreate(&time3); cudaEventCreate(&time4);
    cudaEventCreate(&time2BW); cudaEventCreate(&time2Sobel);
    cudaEventCreate(&time2Gauss);

    cudaEventRecord(time1, 0);
    uint64_t GPUtotalBufferSize = 4 *sizeof(double)*ip.IMAGEPIX + 2 *sizeof(uint8_t)*ip.IMAGESIZE;
    gpuErrchk(cudaMalloc((void**)&GPUptr, GPUtotalBufferSize));
    
    GPUImg       = (uint8_t *)GPUptr;
    GPUResultImg = GPUImg + ip.IMAGESIZE;
    GPUBWImg     = (double *)(GPUResultImg + ip.IMAGESIZE);
    GPUGaussImg  = GPUBWImg + ip.IMAGEPIX;
    GPUGradient  = GPUGaussImg + ip.IMAGEPIX;
    GPUTheta     = GPUGradient + ip.IMAGEPIX;

    gpuErrchk(cudaMemcpy(GPUImg, TheImg, ip.IMAGESIZE, cudaMemcpyHostToDevice));
    cudaEventRecord(time2, 0);

    BlkPerRow=CEIL(IPH, ThrPerBlk);
    NumBlocks=IPV*BlkPerRow;

    BWKernel <<< NumBlocks, ThrPerBlk >>> (GPUBWImg, GPUImg, IPH);
    GPUDataTfrBW =sizeof(double)*ip.IMAGEPIX +sizeof(uint8_t)*ip.IMAGESIZE;
    cudaEventRecord(time2BW, 0);

    GaussKernel <<< NumBlocks, ThrPerBlk >>> (GPUGaussImg, GPUBWImg, IPH, IPV);
    GPUDataTfrGauss = 2*sizeof(double)*ip.IMAGEPIX;
    cudaEventRecord(time2Gauss, 0);// after Gauss image calculation

    SobelKernel <<<  NumBlocks, ThrPerBlk >>> (GPUGradient, GPUTheta, GPUGaussImg, IPH, IPV);
    GPUDataTfrSobel = 3 *sizeof(double)*ip.IMAGEPIX;
    cudaEventRecord(time2Sobel, 0);// after Gradient, Theta computation

    ThresholdKernel <<< NumBlocks, ThrPerBlk >>> (GPUResultImg, GPUGradient,GPUTheta, IPH, IPV, ThreshLo, ThreshHi);
    GPUDataTfrThresh=sizeof(double)*ip.IMAGEPIX +sizeof(uint8_t)*ip.IMAGESIZE;
    cudaEventRecord(time3, 0);  // after threshold
    
    gpuErrchk(cudaMemcpy(CpyImg, GPUResultImg, ip.IMAGESIZE, cudaMemcpyDeviceToHost));
    cudaEventRecord(time4, 0);  // after GPU-> CPU tfr

    
    gpuErrchk(cudaDeviceSynchronize());
    WriteBMPlin(CpyImg, OutputFileName);
    

	cudaEventSynchronize(time1); cudaEventSynchronize(time2);
	cudaEventSynchronize(time3); cudaEventSynchronize(time4);
	cudaEventSynchronize(time2BW); cudaEventSynchronize(time2Gauss);
    cudaEventSynchronize(time2Sobel);
    
    cudaEventElapsedTime(&totalKernelTime, time2, time3);
    cudaEventElapsedTime(&totalTime, time1, time4);
    
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
    cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);
    
    cudaEventElapsedTime(&kernelExecTimeBW, time2, time2BW);
	cudaEventElapsedTime(&kernelExecTimeGauss, time2BW, time2Gauss);
	cudaEventElapsedTime(&kernelExecTimeSobel, time2Gauss, time2Sobel);
	cudaEventElapsedTime(&kernelExecTimeThreshold, time2Sobel, time3);
    
    GPUDataTfrKernel = GPUDataTfrBW + GPUDataTfrGauss + GPUDataTfrSobel + GPUDataTfrThresh;
    GPUDataTfrTotal  = GPUDataTfrKernel + 2 * ip.IMAGESIZE;


    printf("\n\n---------------------\n");
    printf("%s   ComputeCapab=%d.%d [max %s blocks; %d thr/blk] \n",GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
    printf("\n\n---------------------\n");
    printf("%s %s %u %d %d [%u BLOCKS, %u BLOCKS/ROW]\n", InputFileName, OutputFileName, ThrPerBlk, ThreshLo, ThreshHi, NumBlocks,BlkPerRow);
    printf("\n\n---------------------\n");
    printf("CPU->GPU Transfer = %f ms ... %ui MB ... %f GB/s\n", tfrCPUtoGPU, MB(ip.IMAGESIZE),BW(ip.IMAGESIZE,tfrCPUtoGPU));
    printf("GPU->CPU Transfer = %f ms ... %ui MB ... %f GB/s\n", tfrGPUtoCPU, MB(ip.IMAGESIZE),BW(ip.IMAGESIZE, tfrGPUtoCPU));
    printf("\n\n---------------------\n");
    printf("     BW Kernel Execution Time = %f ms ... %lu MB ... %f GB/s\n", kernelExecTimeBW,MB(GPUDataTfrBW), BW(GPUDataTfrBW, kernelExecTimeBW));
    printf("   Gauss Kernel Execution Time = %f ms ... %lu MB ... %f GB/s\n", kernelExecTimeGauss,MB(GPUDataTfrGauss), BW(GPUDataTfrGauss, kernelExecTimeGauss));
    printf("   Sobel Kernel Execution Time = %f ms ... %lu MB ... %f GB/s\n", kernelExecTimeSobel,MB(GPUDataTfrSobel), BW(GPUDataTfrSobel, kernelExecTimeSobel));
    printf("Threshold Kernel Execution Time = %f ms ... %lu MB ... %f GB/s\n", kernelExecTimeThreshold,MB(GPUDataTfrThresh), BW(GPUDataTfrThresh, kernelExecTimeThreshold));
    printf("\n\n---------------------\n");
    printf("       Total Kernel-only time = %f ms ... %lf MB ... %f GB/s\n", totalKernelTime,MB(GPUDataTfrKernel), BW(GPUDataTfrKernel, totalKernelTime));
    printf("  Total time with I/O included = %f ms ... %lf MB ... %f GB/s\n", totalTime, MB(GPUDataTfrTotal),BW(GPUDataTfrTotal, totalTime));
    printf("\n\n---------------------\n");


    return 0;

}