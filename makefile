PTX_ASSEMBLER_OPTIONS=--ptxas-options -O3,-v,-warn-spills
COMPILER_OPTIONS=--compiler-options -W4
GPU_ARCHITECTURE=61
COMPILE_DRIVER_PARAMS= $(DISABLE_ASSERTIONS)  $(COMPILER_OPTIONS) --relocatable-device-code true --cudart static --machine 64 -x cu -O3 $(PTX_ASSEMBLER_OPTIONS) --gpu-architecture=compute_$(GPU_ARCHITECTURE) --gpu-code=sm_$(GPU_ARCHITECTURE),compute_$(GPU_ARCHITECTURE) --use_fast_math
DEPENDENCIES=main.cu assignments.cu
OUTPUT_EXECUTABLE_NAME=out

build $(DEPENDENCIES):
	nvcc $(COMPILE_DRIVER_PARAMS) -o $(OUTPUT_EXECUTABLE_NAME) $(DEPENDENCIES)

clean:
	rm $(OUTPUT_EXECUTABLE_NAME).exe $(OUTPUT_EXECUTABLE_NAME).lib $(OUTPUT_EXECUTABLE_NAME).exp