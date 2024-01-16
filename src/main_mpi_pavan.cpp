#include <string>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "Mapper2D.h"
#include "Parameter.h"
#include "utilities.h"

int main()
{

    Timer timer;

    constexpr SimulationParameter parameter;

    // The fixed number of partitions in x an y direction, e.g. 2x2 = 4
    // The total number of processes has to match the call of mpirun. 
    // In this case: mpirun -np 4 ./Laplace_mpi
    constexpr int numPartsX = 3;
    constexpr int numPartsY = 2;

    constexpr int localNx = parameter.gridNx / numPartsX; //division using integers -> not precise
    constexpr int localNy = parameter.gridNy / numPartsY;

    // -> hence, we calculate the effective numbers again
    constexpr int realGridNx = localNx * numPartsX;
    constexpr int realGridNy = localNy * numPartsY;

    constexpr Mapper2D innerGrid(localNx, localNy);

    /* do parallelisation from here on ... */
    MPI_Init(NULL, NULL);

    int myRank;
    int numProcesses;

    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    /* How to derive the 2D-process number form the 1D-MPI-rank */
    Mapper2D processTopology = Mapper2D(numPartsX, numPartsY);
    int myXRank = processTopology.xForPos(myRank);
    int myYRank = processTopology.yForPos(myRank);

    // The entire grid has a ghost layer on each side.
    constexpr Mapper2D entireGrid(innerGrid.nx() + 2, innerGrid.ny() + 2);

    /* receive buffers for ghost layer data */
    double *leftReceiveBuffer   = new double[innerGrid.ny()];
    double *rightReceiveBuffer  = new double[innerGrid.ny()];
    double *topReceiveBuffer    = new double[innerGrid.nx()];
    double *bottomReceiveBuffer = new double[innerGrid.nx()];

    /* send buffers */
    double *leftSendBuffer = new double[innerGrid.ny()];
    double *rightSendBuffer = new double[innerGrid.ny()];
    double *topSendBuffer = new double[innerGrid.nx()];
    double *bottomSendBuffer = new double[innerGrid.nx()];

    /* Data initialization */
    std::vector<double> oldData (entireGrid.size(),0.0);
    std::vector<double> newData (entireGrid.size(),0.0);

    /* In the parallel version also initialize send buffers here ... */
    for(size_t i = 0; i < innerGrid.ny(); i++)
    {
        leftSendBuffer[i] = 0.0;
        rightSendBuffer[i] = 0.0;
    }

    for (size_t i = 0; i < innerGrid.nx(); i++)
    {
        topSendBuffer[i] = 0.0;
        bottomSendBuffer[i] = 0.0;
    }

    /* In the parallel version the following variables need to be calculated. The name "cell" is an equivalent for process. */
    bool isLeftBoundaryCell = false;
    bool isRightBoundaryCell = false;
    bool isBottomBoundaryCell = false;
    bool isTopBoundaryCell = false;

    if (myXRank == 0)                          
        isLeftBoundaryCell = true;        
    
    if (myXRank == ( numPartsX - 1 ))
        isRightBoundaryCell = true;

    if ( myYRank == ( numPartsY - 1 ))
        isBottomBoundaryCell = true;
    
    if ( myYRank == 0)
        isTopBoundaryCell = true;

    /* set boundary conditions ... */
    if (isLeftBoundaryCell)
        for (size_t i = 0; i < innerGrid.ny(); i++)
            leftReceiveBuffer[i] = parameter.bcLeft;

    if (isRightBoundaryCell)
        for (size_t i = 0; i < innerGrid.ny(); i++)
            rightReceiveBuffer[i] = parameter.bcRight;

    if (isTopBoundaryCell)
        for (size_t i = 0; i < innerGrid.nx(); i++)
            topReceiveBuffer[i] = parameter.bcTop;

    if (isBottomBoundaryCell)
        for (size_t i = 0; i < innerGrid.nx(); i++)
            bottomReceiveBuffer[i] = parameter.bcBottom;

    /* Tag for sending */
    int rightsendtag    = 1;
    int bottomsendtag   = 2;
    int leftsendtag     = 3;
    int topsendtag      = 4;

    /* Tag for recieving*/
    int rightrecvtag    = 3;
    int bottomrecvtag   = 4;
    int leftrecvtag     = 1;
    int toprecvtag      = 2;

    /* Size of messages */
    int countY = (int)innerGrid.ny();
    int countX = (int)innerGrid.nx();

    /* Source/destination rank */
    int rightnode = (int)processTopology.pos( ( myXRank + 1), myYRank);
    int bottomnode = (int)processTopology.pos( myXRank , ( myYRank + 1 ) );
    int leftnode = (int)processTopology.pos( (myXRank - 1) , myYRank );
    int topnode = (int)processTopology.pos( myXRank , ( myYRank - 1 ) );

    size_t numComm; //  Number of Communications

    int iteration = 0;

    timer.startNupsTimer();

    MPI_Request request[8];
    MPI_Status status[8];
    
    /* iteration */
    bool done = false;
    while (!done)
    {
        double error = 0.0;
        double diff;
        /* in the parallel version: Do the send and receive here. Prefer doing it in the background (nonblocking / async). Watch out for deadlocks! */
        /* ...TODO: Add communication here... */

        if (myXRank == 0 && myYRank == 0) // top left node
        {
            numComm = 4;
            MPI_Isend(rightSendBuffer,  countY, MPI_DOUBLE, rightnode,  rightsendtag ,   MPI_COMM_WORLD, &request[0]);
            MPI_Isend(bottomSendBuffer, countX, MPI_DOUBLE, bottomnode, bottomsendtag ,  MPI_COMM_WORLD, &request[1]);

            MPI_Irecv(rightReceiveBuffer,  countY, MPI_DOUBLE, rightnode,  rightrecvtag ,  MPI_COMM_WORLD, &request[2]);
            MPI_Irecv(bottomReceiveBuffer, countX, MPI_DOUBLE, bottomnode, bottomrecvtag , MPI_COMM_WORLD, &request[3]);
        }
        else if (myXRank == (int)(processTopology.nx()-1) && myYRank == 0) // top right node
        {
            numComm = 4;
            MPI_Isend(bottomSendBuffer, countX, MPI_DOUBLE, bottomnode,  bottomsendtag ,  MPI_COMM_WORLD, &request[0]);
            MPI_Isend(leftSendBuffer,   countY, MPI_DOUBLE, leftnode,    leftsendtag ,    MPI_COMM_WORLD, &request[1]);

            MPI_Irecv(bottomReceiveBuffer, countX, MPI_DOUBLE, bottomnode, bottomrecvtag , MPI_COMM_WORLD, &request[2]);
            MPI_Irecv(leftReceiveBuffer,   countY, MPI_DOUBLE, leftnode,   leftrecvtag ,   MPI_COMM_WORLD, &request[3]);
        }
        else if (myXRank == 0 && myYRank == (int)(processTopology.ny()-1)) // bottom left node
        {
            numComm = 4;
            MPI_Isend(topSendBuffer,   countX, MPI_DOUBLE, topnode,   topsendtag ,   MPI_COMM_WORLD, &request[0]);
            MPI_Isend(rightSendBuffer, countY, MPI_DOUBLE, rightnode, rightsendtag , MPI_COMM_WORLD, &request[1]);

            MPI_Irecv(topReceiveBuffer,   countX, MPI_DOUBLE, topnode,   toprecvtag ,  MPI_COMM_WORLD, &request[2]);
            MPI_Irecv(rightReceiveBuffer, countY, MPI_DOUBLE, rightnode, rightrecvtag , MPI_COMM_WORLD, &request[3]);
        }
        else if (myXRank == (int)(processTopology.nx()-1) && myYRank == (int)(processTopology.ny()-1)) // bottom right node
        {
            numComm = 4;
            MPI_Isend(leftSendBuffer, countY, MPI_DOUBLE, leftnode, leftsendtag , MPI_COMM_WORLD, &request[0]);
            MPI_Isend(topSendBuffer,  countX, MPI_DOUBLE, topnode,  topsendtag ,  MPI_COMM_WORLD, &request[1]);

            MPI_Irecv(leftReceiveBuffer, countY, MPI_DOUBLE, leftnode, leftrecvtag , MPI_COMM_WORLD, &request[2]);
            MPI_Irecv(topReceiveBuffer,  countX, MPI_DOUBLE, topnode,  toprecvtag ,  MPI_COMM_WORLD, &request[3]);            
        }
        else if (myXRank > 0 && myXRank < (int)(processTopology.nx()-1) && myYRank == 0) // top row nodes
        {
            numComm = 6;
            MPI_Isend(rightSendBuffer,  countY, MPI_DOUBLE, rightnode,  rightsendtag ,  MPI_COMM_WORLD, &request[0]);
            MPI_Isend(bottomSendBuffer, countX, MPI_DOUBLE, bottomnode, bottomsendtag , MPI_COMM_WORLD, &request[1]);
            MPI_Isend(leftSendBuffer,   countY, MPI_DOUBLE, leftnode,   leftsendtag ,   MPI_COMM_WORLD, &request[2]);

            MPI_Irecv(rightReceiveBuffer,  countY, MPI_DOUBLE, rightnode,  rightrecvtag ,  MPI_COMM_WORLD, &request[3]);
            MPI_Irecv(bottomReceiveBuffer, countX, MPI_DOUBLE, bottomnode, bottomrecvtag , MPI_COMM_WORLD, &request[4]);
            MPI_Irecv(leftReceiveBuffer,   countY, MPI_DOUBLE, leftnode,   leftrecvtag ,   MPI_COMM_WORLD, &request[5]);
        }
        else if (myXRank > 0 && myXRank < (int)(processTopology.nx()-1) && myYRank == (int)(processTopology.ny()-1) ) // bottom row nodes
        {
            numComm = 6;
            MPI_Isend(rightSendBuffer, countY, MPI_DOUBLE, rightnode, rightsendtag , MPI_COMM_WORLD, &request[0]);
            MPI_Isend(leftSendBuffer,  countY, MPI_DOUBLE, leftnode,  leftsendtag ,  MPI_COMM_WORLD, &request[1]);
            MPI_Isend(topSendBuffer,   countX, MPI_DOUBLE, topnode,   topsendtag ,   MPI_COMM_WORLD, &request[2]);

            MPI_Irecv(rightReceiveBuffer, countY, MPI_DOUBLE, rightnode, rightrecvtag , MPI_COMM_WORLD, &request[3]);
            MPI_Irecv(leftReceiveBuffer,  countY, MPI_DOUBLE, leftnode,  leftrecvtag ,  MPI_COMM_WORLD, &request[4]);
            MPI_Irecv(topReceiveBuffer,   countX, MPI_DOUBLE, topnode,   toprecvtag ,   MPI_COMM_WORLD, &request[5]); 
        }
        else if (myXRank == 0 && myYRank > 0 && myYRank < (int)(processTopology.ny()-1)) // left column nodes
        {
            numComm = 6;
            MPI_Isend(rightSendBuffer,  countY, MPI_DOUBLE, rightnode,  rightsendtag ,  MPI_COMM_WORLD, &request[0]);
            MPI_Isend(bottomSendBuffer, countX, MPI_DOUBLE, bottomnode, bottomsendtag , MPI_COMM_WORLD, &request[1]);
            MPI_Isend(topSendBuffer,    countX, MPI_DOUBLE, topnode,    topsendtag ,    MPI_COMM_WORLD, &request[2]);

            MPI_Irecv(rightReceiveBuffer,  countY, MPI_DOUBLE, rightnode,  rightrecvtag ,  MPI_COMM_WORLD, &request[3]);
            MPI_Irecv(bottomReceiveBuffer, countX, MPI_DOUBLE, bottomnode, bottomrecvtag , MPI_COMM_WORLD, &request[4]);
            MPI_Irecv(topReceiveBuffer,    countX, MPI_DOUBLE, topnode,    toprecvtag ,    MPI_COMM_WORLD, &request[5]);
        }
        else if (myXRank == (int)(processTopology.nx()-1) && myYRank > 0 && myYRank < (int)(processTopology.ny()-1)) // right column nodes
        {
            numComm = 6;
            MPI_Isend(bottomSendBuffer, countX, MPI_DOUBLE, bottomnode, bottomsendtag , MPI_COMM_WORLD, &request[0]);
            MPI_Isend(leftSendBuffer,   countY, MPI_DOUBLE, leftnode,   leftsendtag ,   MPI_COMM_WORLD, &request[1]);
            MPI_Isend(topSendBuffer,    countX, MPI_DOUBLE, topnode,    topsendtag ,    MPI_COMM_WORLD, &request[2]);

            MPI_Irecv(bottomReceiveBuffer, countX, MPI_DOUBLE, bottomnode, bottomrecvtag , MPI_COMM_WORLD, &request[3]);
            MPI_Irecv(leftReceiveBuffer,   countY, MPI_DOUBLE, leftnode,   leftrecvtag ,   MPI_COMM_WORLD, &request[4]);
            MPI_Irecv(topReceiveBuffer,    countX, MPI_DOUBLE, topnode,    toprecvtag ,    MPI_COMM_WORLD, &request[5]);
        }
        else  // Inner nodes
        {
            numComm = 8;
            MPI_Isend(rightSendBuffer,  countY, MPI_DOUBLE, rightnode,  rightsendtag ,  MPI_COMM_WORLD, &request[0]);
            MPI_Isend(bottomSendBuffer, countX, MPI_DOUBLE, bottomnode, bottomsendtag , MPI_COMM_WORLD, &request[1]);
            MPI_Isend(leftSendBuffer,   countY, MPI_DOUBLE, leftnode,   leftsendtag ,   MPI_COMM_WORLD, &request[2]);
            MPI_Isend(topSendBuffer,    countX, MPI_DOUBLE, topnode,    topsendtag ,    MPI_COMM_WORLD, &request[3]);

            MPI_Irecv(rightReceiveBuffer,  countY, MPI_DOUBLE, rightnode,  rightrecvtag ,  MPI_COMM_WORLD, &request[4]);
            MPI_Irecv(bottomReceiveBuffer, countX, MPI_DOUBLE, bottomnode, bottomrecvtag , MPI_COMM_WORLD, &request[5]);
            MPI_Irecv(leftReceiveBuffer,   countY, MPI_DOUBLE, leftnode,   leftrecvtag ,   MPI_COMM_WORLD, &request[6]);
            MPI_Irecv(topReceiveBuffer,    countX, MPI_DOUBLE, topnode,    toprecvtag ,    MPI_COMM_WORLD, &request[7]);
        }
        

        /* first do the calculations without the ghost layers */
        for (size_t y = 2; y < entireGrid.ny() - 2; y++)
            for (size_t x = 2; x < entireGrid.nx() - 2; x++)
            {
                newData[entireGrid.pos(x, y)] = 0.25 * (oldData[entireGrid.pos(x - 1, y)] +
                                                        oldData[entireGrid.pos(x + 1, y)] +
                                                        oldData[entireGrid.pos(x, y - 1)] +
                                                        oldData[entireGrid.pos(x, y + 1)]);
                diff = newData[entireGrid.pos(x, y)] - oldData[entireGrid.pos(x, y)];
                error = error + diff * diff;
            }

        /* check if all the communication has taken place before inserting the ghost layers*/
        
        for (size_t i = 0; i < numComm; i++)
        {
            int flag = 0;
            while (!flag)
            {
                MPI_Test(&request[i], &flag, &status[i]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        /* now ghost layers should have been received ... */
        /* insert ghost layers */
        for (size_t x = 1; x < entireGrid.nx() - 1; x++)
        {
            oldData[entireGrid.pos(x, 0)] = topReceiveBuffer[x - 1];
            oldData[entireGrid.pos(x, entireGrid.ny() - 1)] = bottomReceiveBuffer[x - 1];
        }

        for (size_t y = 1; y < entireGrid.ny() - 1; y++)
        {
            oldData[entireGrid.pos(0, y)] = leftReceiveBuffer[y - 1];
            oldData[entireGrid.pos(entireGrid.nx() - 1, y)] = rightReceiveBuffer[y - 1];
        }

        /* Now do the rest of the calculation including the ghost layers. */
        
        for (size_t x = 1; x < entireGrid.nx() - 1; x++)
        {
            // top
            newData[entireGrid.pos(x, 1)] = 0.25 * (oldData[entireGrid.pos(x - 1, 1)] +
                                                    oldData[entireGrid.pos(x + 1, 1)] +
                                                    oldData[entireGrid.pos(x, 0)] +
                                                    oldData[entireGrid.pos(x, 2)]);
            diff = newData[entireGrid.pos(x, 1)] - oldData[entireGrid.pos(x, 1)];
            error = error + diff * diff;

            topSendBuffer[x-1] = newData[entireGrid.pos(x,1)];

            // bottom
            newData[entireGrid.pos(x, entireGrid.ny() - 2)] = 0.25 * (oldData[entireGrid.pos(x - 1, entireGrid.ny() - 2)] +
                                                                         oldData[entireGrid.pos(x + 1, entireGrid.ny() - 2)] +
                                                                         oldData[entireGrid.pos(x, entireGrid.ny() - 3)] +
                                                                         oldData[entireGrid.pos(x, entireGrid.ny() - 1)]);
            diff = newData[entireGrid.pos(x, entireGrid.ny() - 2)] - oldData[entireGrid.pos(x, entireGrid.ny() - 2)];
            error = error + diff * diff;

            bottomSendBuffer[x-1] = newData[entireGrid.pos(x, entireGrid.ny() - 2)];
        }

        for (size_t y = 1; y < entireGrid.ny() - 1; y++)
        {
            // left
            newData[entireGrid.pos(1, y)] = 0.25 * (oldData[entireGrid.pos(1, y - 1)] +
                                                    oldData[entireGrid.pos(1, y + 1)] +
                                                    oldData[entireGrid.pos(0, y)] +
                                                    oldData[entireGrid.pos(2, y)]);
            diff = newData[entireGrid.pos(1, y)] - oldData[entireGrid.pos(1, y)];
 
            leftSendBuffer[y-1] = newData[entireGrid.pos(1, y)];

            // right
            newData[entireGrid.pos(entireGrid.nx() - 2, y)] = 0.25 * (oldData[entireGrid.pos(entireGrid.nx() - 3, y)] +
                                                                         oldData[entireGrid.pos(entireGrid.nx() - 1, y)] +
                                                                         oldData[entireGrid.pos(entireGrid.nx() - 2, y - 1)] +
                                                                         oldData[entireGrid.pos(entireGrid.nx() - 2, y + 1)]);
            diff = newData[entireGrid.pos(entireGrid.nx() - 2, y)] - oldData[entireGrid.pos(entireGrid.nx() - 2, y)];
            error = error + diff * diff;

            rightSendBuffer[y-1] = newData[entireGrid.pos(entireGrid.nx()-2, y)];
        }

        std::swap(oldData, newData);

        /* Stop in case of little change */
        // in parallel case: collect the error from the other (Allreduce)
        double totalError = 0;
        MPI_Allreduce(&error, &totalError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        done = (totalError < 1.0e-4);

        iteration++;
        if ((iteration % parameter.outputInterval) == 0)
        {
            const auto mnups = timer.getMNups(innerGrid.size() * parameter.outputInterval);
            std::cout << "time step: " << iteration << " error: " << error << " MNUPS: " << mnups << "\n";

            timer.startNupsTimer();
        }
    }

    /* Output (Only process 0. In the parallel case process 0 needs to collect the necessary data for the output from the other processes. */

    if (myRank == 0)
    {
        MPI_Status status1;
        double *resultData = new double[realGridNx * realGridNy];
        Mapper2D globalGrid = Mapper2D(realGridNx, realGridNy);
        for (size_t x = 1; x < entireGrid.nx() - 1; x++)
            for (size_t y = 1; y < entireGrid.ny() - 1; y++)
                resultData[globalGrid.pos(x - 1, y - 1)] = oldData[entireGrid.pos(x, y)];

        for (int partX = 0; partX < numPartsX; partX++)
            for (int partY = 0; partY < numPartsY; partY++)
                if (partX || partY) //if (!(i==0 && j==0)
                {
                    std::cout << "Partition X = " << partX << ", Partition Y = " << partY << std::endl;
                    for (size_t y = 0; y < entireGrid.ny() - 2; y++) // line by line
                        MPI_Recv(resultData + globalGrid.pos(partX * localNx, partY * localNy + y), entireGrid.nx() - 2, MPI_DOUBLE, processTopology.pos(partX, partY), 0, MPI_COMM_WORLD, &status1);
                }
        writeUCDFile(parameter.outputFileName, resultData, globalGrid);
        delete[] resultData;
    }
    else
    {
        for (size_t y = 1; y < entireGrid.ny() - 1; y++) // line by line
            MPI_Ssend(oldData.data() + entireGrid.pos(1, y), entireGrid.nx() - 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    const auto runtime = timer.getRuntimeSeconds();
    std::cout << "Runtime: " << runtime << " s. " << std::endl;

    delete[] leftReceiveBuffer;
    delete[] rightReceiveBuffer;
    delete[] topReceiveBuffer;
    delete[] bottomReceiveBuffer;

    delete[] leftSendBuffer;
    delete[] rightSendBuffer;
    delete[] topSendBuffer;
    delete[] bottomSendBuffer;

    MPI_Finalize();

    return 0;
}
