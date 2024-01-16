#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>

#include "Mapper2D.h"
#include "Parameter.h"
#include "utilities.h"

void calculateTimestep(const std::vector<double>& oldData, std::vector<double>& newData, Mapper2D innerGrid, Mapper2D entireGrid, size_t numThreads, size_t threadID)
{
    size_t x = 0;
    size_t y = 0;

    //for (size_t i = 0 + numThreads * threadID; i < innerGrid.size(); i++) // slow due to cache misses
    for (size_t i = threadID; i < innerGrid.size(); i = i + numThreads) 
    {
        x = innerGrid.xForPos(i) + 1;
        y = innerGrid.yForPos(i) + 1;

        newData[entireGrid.pos(x, y)] = 0.25 * (oldData[entireGrid.pos(x - 1, y)] +
                                                oldData[entireGrid.pos(x + 1, y)] +
                                                oldData[entireGrid.pos(x, y - 1)] +
                                                oldData[entireGrid.pos(x, y + 1)]);
    }
}

int main(int argc, char **argv)
{
    size_t numThreads = 1;
    if(argc == 2)
        numThreads = atoi(argv[1]);

    Timer timer;

    constexpr SimulationParameter parameter;

    // make sure we can easily distribute the work among all threads
    size_t effectiveNx = (parameter.gridNx + numThreads - 1) / numThreads * numThreads;
    size_t effectiveNy = (parameter.gridNy + numThreads - 1) / numThreads * numThreads;

    Mapper2D innerGrid(effectiveNx, effectiveNy);

    std::cout << "Number of threads: " << numThreads << std::endl;
    std::cout << "Grid size: " << innerGrid.nx() << "x" << innerGrid.ny() << std::endl;

    // The entire grid has a ghost layer on each side.
    Mapper2D entireGrid(innerGrid.nx() + 2, innerGrid.ny() + 2);

    // we hold the actual data twice so we can easily apply our algorithm.
    std::vector<double> oldData (entireGrid.size());
    std::vector<double> newData (entireGrid.size());

    /* initialization */
    for (size_t i = 0; i < entireGrid.size(); i++)
    {
        oldData[i] = 0.0;
        newData[i] = 0.0;
    }

    // BC initialization
    for (size_t y = 0; y < entireGrid.ny(); y++)
    {
        oldData[entireGrid.pos(0, y)] = parameter.bcLeft;
        oldData[entireGrid.pos(entireGrid.nx() - 1, y)] = parameter.bcRight;

        newData[entireGrid.pos(0, y)] = parameter.bcLeft;
        newData[entireGrid.pos(entireGrid.nx() - 1, y)] = parameter.bcRight;
    }

    for (size_t x = 0; x < entireGrid.nx(); x++)
    {
        oldData[entireGrid.pos(x, 0)] = parameter.bcTop;
        oldData[entireGrid.pos(x, entireGrid.ny() - 1)] = parameter.bcBottom;

        newData[entireGrid.pos(x, 0)] = parameter.bcTop;
        newData[entireGrid.pos(x, entireGrid.ny() - 1)] = parameter.bcBottom;
    }

    std::vector<std::thread> threads(numThreads);

    int iteration = 0;
    timer.startNupsTimer();
    
    /* iteration */
    bool done = false;
    while (!done)
    {
        for (size_t i = 0; i < numThreads; i++)
        {
            threads[i] = std::thread(calculateTimestep, std::ref(oldData), std::ref(newData), innerGrid, entireGrid, numThreads, i);
        }

        for (auto& thread : threads)
        {
            thread.join();
        }

        iteration++;
        if ((iteration % parameter.outputInterval) == 0)
        {
            const auto mnups = timer.getMNups(long(innerGrid.size() * parameter.outputInterval));

            /* stop in case of little change */
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

    return 0;
}
