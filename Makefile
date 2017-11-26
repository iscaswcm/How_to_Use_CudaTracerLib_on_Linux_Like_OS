ECHO       := echo -e
DEF        := -m64  -DUSE_CUDA
BUILD      := build.debug
LIB        := -L../$(BUILD) \
	          -L/usr/local/cuda-9.0/lib64 \
			  -lCudaTracerLib \
			  -lboost_system \
			  -lboost_filesystem \
			  -lboost_program_options \
			  -lboost_iostreams \
			  -lcudart
INCLUDE    := -I. -Isrc -Isrc/include -I../CudaTracerLib.ori -I../qMatrixLib  -I/usr/local/cuda-9.0/include
CXX_FLAGS  := -std=c++11  $(INCLUDE) 

CXX        := g++ $(DEF)
rwildcard   = $(foreach d, $(wildcard $1*), $(call rwildcard,$d/,$2)$(filter $(subst *,%,$2),$d))

#$(wildcard $1*) -- get all filenane directories in current directory

CXX_SOURCES := $(call rwildcard, src/, *.cpp)  
CXX_OBJS    := $(subst src/, objs/, $(CXX_SOURCES:.cpp=.cpp.o))

TARGET       := mitsubacuda

$(TARGET): $(CXX_OBJS) $(CU_OBJS) 
	@$(ECHO) "\033[32;49;6mLinking $@\033[0m"
	$(CXX)  -o $@ $^ $(CXX_FLAGS) $(LIB)
	@cp src/misc/cudatracerlib.ini .

objs/%.cpp.o: src/%.cpp
	@mkdir -p $(@D)
	@$(ECHO) "\033[32;49;1mCompiling $< to $@\033[0m"
	@$(CXX) $(CXX_FLAGS) -o $@ -c $<

clean:
	@$(ECHO) $(CXX_SOURCES) $(CU_SOURCES)
	@$(ECHO) $(CXX_OBJS) $(CU_OBJS)
	@rm -rf   objs

