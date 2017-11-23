ECHO       := echo -e
DEF        := -m64  -DUSE_CUDA
BUILD      := build.debug
LIB        := -L../$(BUILD) -lCudaTracerLib -lboost_system -lboost_filesystem -lboost_program_options -lboost_iostreams -lfreeimage
INCLUDE    := -I. -Isrc -Isrc/include -I../CudaTracerLib.ori -I../qMatrixLib  
NVCC_FLAGS := -std=c++11 -arch=compute_35 -code=sm_35 --relocatable-device-code true $(INCLUDE) 

NVCC       := nvcc $(DEF)
rwildcard   = $(foreach d, $(wildcard $1*), $(call rwildcard,$d/,$2)$(filter $(subst *,%,$2),$d))

#$(wildcard $1*) -- get all filenane directories in current directory

NVCC_SOURCES := $(call rwildcard, src/, *.cpp)  
NVCC_OBJS    := $(subst src/, objs/, $(NVCC_SOURCES:.cpp=.cpp.o))

CU_SOURCES   := $(call rwildcard, src/, *.cu)  
CU_OBJS      := $(subst src/, objs/,$(CU_SOURCES:.cu=.cu.o))

#TARGET       := mitsubacuda
TARGET       := hostExample

$(TARGET): $(NVCC_OBJS) $(CU_OBJS) 
	@$(ECHO) "\033[32;49;6mLinking $@\033[0m"
	$(NVCC)  -o $@ $^ $(NVCC_FLAGS) $(LIB)
	@cp src/misc/cudatracerlib.ini .

objs/%.cpp.o: src/%.cpp
	@mkdir -p $(@D)
	@$(ECHO) "\033[32;49;1mCompiling $< to $@\033[0m"
	@$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

objs/%.cu.o: src/%.cu
	@mkdir -p $(@D)
	@$(ECHO) "\033[32;49;1mCompiling $< to $@\033[0m"
	@$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

clean:
	@$(ECHO) $(NVCC_SOURCES) $(CU_SOURCES)
	@$(ECHO) $(NVCC_OBJS) $(CU_OBJS)
	@rm -rf   objs

