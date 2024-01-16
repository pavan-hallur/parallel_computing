cmake_minimum_required(VERSION 3.15)

project(Laplace2D CXX)

set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_BUILD_TYPE RelWithDebInfo)

enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_library(compiler_flags INTERFACE)

if(UNIX)
    target_compile_options(compiler_flags INTERFACE -Wall -Wextra -pedantic -Werror -O1)

    # code coverage gcov
    if (ENABLE_COVERAGE AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        message(STATUS "Code coverage enabled.")
        target_compile_options(compiler_flags INTERFACE --coverage)
        target_link_options(compiler_flags INTERFACE --coverage)
    endif()

elseif(MSVC)
    target_compile_options(compiler_flags INTERFACE /W4 /WX)
endif()



add_subdirectory(src)

add_subdirectory(tests)
