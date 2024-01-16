#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) 
        exit(code);
   }
}


void setDevice(int device)
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    if(device < num_devices)
    {
        cudaSetDevice(device);
    } 
    else
    {
        printf("Cannot set device: %d.\n", device);
        return exit(1);
    }
    
    printf("Number of devices: %d\nDevice set: %d\n", num_devices, device);
}

#endif
