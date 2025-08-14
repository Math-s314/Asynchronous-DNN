cmake_minimum_required(VERSION 3.30)
project(OpenCLTest)

set(CMAKE_CXX_STANDARD 17)

include_directories(inc)
unset(OpenCL_DIR CACHE)
find_package(OpenCL PATHS "$ENV{OpenCLDir}" NO_DEFAULT_PATH CONFIG REQUIRED)

add_executable(OpenCLTest
    inc/matrix.hpp
    inc/square_matrix.hpp
    inc/vector.hpp
    src/main.cpp
    src/matrix.cpp
    src/square_matrix.cpp
    src/vector.cpp
    inc/base.hpp
    src/base.cpp
)

target_link_libraries(OpenCLTest OpenCL::OpenCL)
target_link_libraries(OpenCLTest OpenCL::UtilsCpp)
target_compile_definitions(OpenCLTest
    PUBLIC
        CL_HPP_ENABLE_EXCEPTIONS
        CL_HPP_TARGET_OPENCL_VERSION=210)
