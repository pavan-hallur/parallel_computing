#include <string>
#include <iostream>
#include <math.h>
#include <chrono>

#include <cuda_runtime.h>

#include "Mapper2D.h"
#include "Parameter.h"
#include "utilities.h"
#include "CudaHelper.h"


__global__ void calculateTimestep(double* oldData, double* newData, Mapper2D innerGrid, Mapper2D entireGrid)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    const int x = innerGrid.xForPos(index) + 1; // add offset of 1 for entire grid
    const int y = innerGrid.yForPos(index) + 1; // add offset of 1 for entire grid

    const double neighborTop    = oldData[entireGrid.pos(x, y - 1)];
    const double neighborBottom = oldData[entireGrid.pos(x, y + 1)];
    const double neighborLeft   = oldData[entireGrid.pos(x - 1, y)];
    const double neighborRight  = oldData[entireGrid.pos(x + 1, y)];

    newData[entireGrid.pos(x, y)] = 0.25 * (neighborTop + neighborBottom + neighborLeft + neighborRight);
}

int main(int argc, char **argv)
{
    int device = 0;
    if(argc == 2)
        device = atoi(argv[1]);
    setDevice(device);

    Timer timer;

    constexpr SimulationParameter parameter;

    // Both gridNx and gridNy must be a multiple of 16 due to the warp/block size 
    if (0 != parameter.gridNx % 16 || 0 != parameter.gridNy % 16) {
        std::cout << "gridNx and gridNy must be a multiple of 16!" << std::endl;
    } 

    constexpr Mapper2D innerGrid(parameter.gridNx, parameter.gridNy);

    // The entire grid includes a node for the boundary condrition on each side.
    constexpr Mapper2D entireGrid(parameter.gridNx + 2, parameter.gridNy + 2);

    double *oldData;
    double *newData;

    // allocating unified memory, which is accesible from the CPU and the GPU
    // https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
    cudaMallocManaged(&oldData, entireGrid.size() * sizeof(double));
    cudaMallocManaged(&newData, entireGrid.size() * sizeof(double));
    gpuErrorCheck( cudaPeekAtLastError() );

    // initialization
    for (int i = 0; i < entireGrid.size(); i++)
    {
        oldData[i] = 0.0;
        newData[i] = 0.0;
    }

    // BC initialization
    for (int i = 0; i < entireGrid.nx(); i++)
    {
        oldData[entireGrid.pos(0, i)] = parameter.bcLeft;
        oldData[entireGrid.pos(entireGrid.nx() - 1, i)] = parameter.bcRight;

        newData[entireGrid.pos(0, i)] = parameter.bcLeft;
        newData[entireGrid.pos(entireGrid.nx() - 1, i)] = parameter.bcRight;
    }

    for (int i = 0; i < entireGrid.ny(); i++)
    {
        oldData[entireGrid.pos(i, 0)] = parameter.bcTop;
        oldData[entireGrid.pos(i, entireGrid.ny() - 1)] = parameter.bcBottom;

        newData[entireGrid.pos(i, 0)] = parameter.bcTop;
        newData[entireGrid.pos(i, entireGrid.ny() - 1)] = parameter.bcBottom;
    }

    const int blockSize = 256;
    const int numBlocks = innerGrid.size() / blockSize;
    printf("Grid: %d, %d\n", numBlocks, blockSize);

    int iteration = 0;
    bool done = false;
    
    timer.startNupsTimer();
    while (!done)
    {
        // calling cuda kernel
        calculateTimestep<<<numBlocks,blockSize>>>(oldData, newData, innerGrid, entireGrid);
        gpuErrorCheck( cudaPeekAtLastError() );

        // checking error every n timesteps
        iteration++;
        if((iteration % parameter.outputInterval) == 0)
        {
            // synchronize: the host waits until the device has completed all preceding requested tasks
            gpuErrorCheck( cudaDeviceSynchronize() );
            const auto mnups = timer.getMNups(innerGrid.size() * parameter.outputInterval);

            // stop in case of little change
            const auto error = calcError(oldData, newData, entireGrid);
            done = (error < 1.0e-4);
            std::cout << "time step: " << iteration << " error: " << error << " MNUPS: " << mnups << "\n";

            timer.startNupsTimer();
        }

        std::swap(oldData, newData);
    }

    const auto runtime = timer.getRuntimeSeconds();
    std::cout << "Runtime: " << runtime << " s. " << std::endl;
    std::cout << "Average MNUPS:" << timer.getAverageMNups(innerGrid.size() * iteration) << std::endl;

    writeUCDFile(parameter.outputFileName, oldData, entireGrid);

    cudaFree(oldData);
    cudaFree(newData);

    return 0;
}
