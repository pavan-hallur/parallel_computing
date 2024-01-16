#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

#include "Mapper2D.h"
#include "Parameter.h"
#include "utilities.h"

int main()
{
    Timer timer;

    constexpr SimulationParameter parameter;

    constexpr Mapper2D innerGrid(parameter.gridNx, parameter.gridNy);

    std::cout << "Grid size: " << innerGrid.nx() << "x" << innerGrid.ny() << std::endl;

    // The entire grid has a ghost layer on each side.
    constexpr Mapper2D entireGrid(innerGrid.nx() + 2, innerGrid.ny() + 2);

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

    int iteration = 0;
    size_t x = 0;
    size_t y = 0;
    
    timer.startNupsTimer();
    
    /* iteration */
    bool done = false;
    while (!done)
    {
        #pragma omp parallel for default(shared) private(x,y) // see 
        for (int i = 0; i < (int)innerGrid.size(); i++) 
        {
            x = innerGrid.xForPos(i) + 1;
            y = innerGrid.yForPos(i) + 1;

            newData[entireGrid.pos(x, y)] = 0.25 * (oldData[entireGrid.pos(x - 1, y)] +
                                                    oldData[entireGrid.pos(x + 1, y)] +
                                                    oldData[entireGrid.pos(x, y - 1)] +
                                                    oldData[entireGrid.pos(x, y + 1)]);
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

        std::swap(newData, oldData);
    }

    const auto runtime = timer.getRuntimeSeconds();
    std::cout << "Runtime: " << runtime << " s. " << std::endl;
    std::cout << "Average MNUPS:" << timer.getAverageMNups(innerGrid.size() * iteration) << std::endl;

    writeUCDFile(parameter.outputFileName, oldData, entireGrid);

    return 0;
}
