

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







	// 	}
	// 	std::cout << std::endl;
	// }
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