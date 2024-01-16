# Serial
add_executable(Laplace2D_serial main_serial.cpp)
target_link_libraries(Laplace2D_serial PUBLIC compiler_flags)
message(STATUS "Serial example added.")


# Multi Threading
add_executable(Laplace2D_threads main_threads.cpp)
target_link_libraries(Laplace2D_threads PUBLIC compiler_flags)

message(STATUS "Multi threading example added.")
find_package(Threads)
if(Threads_FOUND)
    target_link_libraries(Laplace2D_threads PUBLIC Threads::Threads)
endif()

# OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    add_executable(Laplace2D_openmp main_openmp.cpp)
    target_link_libraries(Laplace2D_openmp PUBLIC compiler_flags OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP example added.")
else()
    message(STATUS "OpenMP not found. Skip example.")
endif()


# MPI Parallel
find_package(MPI)
if(MPI_FOUND)
    add_executable(Laplace2D_mpi_blocking main_mpi_pavan.cpp)
    target_link_libraries(Laplace2D_mpi_blocking PUBLIC compiler_flags MPI::MPI_CXX)

    # add_executable(Laplace2D_mpi_non_blocking main_mpi_non_blocking.cpp)
    # target_link_libraries(Laplace2D_mpi_non_blocking PUBLIC compiler_flags MPI::MPI_CXX)

    message(STATUS "MPI example added.")
else()
    message(STATUS "MPI not found. Skip example.")
endif()


# CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    set(CMAKE_CUDA_ARCHITECTURES 60) # phoenix
    add_executable(Laplace2D_cuda main_cuda.cu)

    message(STATUS "CUDA example added.")
else()
    message(STATUS "CUDA not found. Skip example.")
endif()
