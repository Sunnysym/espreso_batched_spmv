ROCM_PATH?= $(wildcard /opt/rocm)
ifeq (,$(ROCM_PATH))
        ROCM_PATH=
endif

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
        HIP_PATH=
endif
HIPCXX=$(HIP_PATH)/bin/hipcc

ifeq (,$(ROCSPARSE_PATH))
        ROCSPARSE_PATH= $(wildcard /opt/rocm/include)
endif

INC = -isystem$(ROCSPARSE_PATH)/include -isystem$(ROCM_PATH)/include -I/public/software/compiler/dtk/dtk-22.10/rocsparse/include -I/usr/lib/gcc/x86_64-redhat-linux/4.8.2/include
HIPCC=$(HIP_PATH)/bin/hipcc
CXXFLAGS = -fopenmp=libgomp -pthread -g -O0 $(INC) 
LDFLAGS = -fopenmp=libgomp -pthread -L$(ROCSPARSE_PATH)/lib -L$(ROCM_PATH)/lib -lrocsparse -Wl,-rpath=$(ROCSPARSE_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib -lm -L/public/software/compiler/dtk/dtk-22.10/rocsparse/lib -L/OPENMP_RUNTIME_LIB
ifneq ($(CXX),$(HIPCXX))
	LDFLAGS += -lamdhip64 
endif

CXX=$(HIPCC)

SOURCES = $(wildcard test_csrspmv_batch.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./256batch_kernel_100

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ -o $@

$(OBJECTS): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf *.o *.out
