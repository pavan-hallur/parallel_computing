include(FetchContent)
message(STATUS "Fetching Catch2: v2.13.4")
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v2.13.4
)

FetchContent_MakeAvailable(Catch2)

add_executable(Laplace2D_tests test_main.cpp test_Mapper2D.cpp)
target_link_libraries(Laplace2D_tests PUBLIC Catch2::Catch2 compiler_flags)
target_include_directories(Laplace2D_tests PUBLIC ${CMAKE_SOURCE_DIR}/src)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(CTest)
include(Catch)
catch_discover_tests(Laplace2D_tests)

message(STATUS "Test exeutable added.")
