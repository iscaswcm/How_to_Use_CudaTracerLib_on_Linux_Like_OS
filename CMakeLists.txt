cmake_minimum_required (VERSION 2.8)

project (CppExample)

set (CMAKE_CXX_STANDARD 11)

find_package( Boost COMPONENTS system filesystem program_options iostreams REQUIRED )
find_package(CUDA REQUIRED)

#set cudatracer library path properly
include_directories(../CudaTracerLib.ori)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(src)
include_directories(src/include)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -m64")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -m64 -std=c++11")
SET(CMAKE_EXE_LINKER_FLAGS "-m64")

file(GLOB CORE_SRCS 
	${PROJECT_SOURCE_DIR}/src/core/*.cpp)
file(GLOB MAIN_SRCS 
	${PROJECT_SOURCE_DIR}/src/main/*.cpp)

add_executable (mitsubacuda ${CORE_SRCS} ${MAIN_SRCS})

find_library(CUDATRACERLIB_PATH CudaTracerLib ../build.debug)
if(NOT CUDATRACERLIB_PATH)
	message(fatal_error "libCudaTracerLib not found")
endif(NOT CUDATRACERLIB_PATH)

message(status   ${CUDATRACERLIB_PATH} " found")

target_link_libraries(mitsubacuda ${Boost_LIBRARIES})
target_link_libraries(mitsubacuda ${CUDA_LIBRARIES})
target_link_libraries(mitsubacuda ${CUDATRACERLIB_PATH})


