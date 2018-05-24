################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../source/ccMDP_GPU.cu \
../source/ccMDP_GPU_modular.cu 

CPP_SRCS += \
../source/cMDPSquareManager.cpp \
../source/main.cpp 

OBJS += \
./source/cMDPSquareManager.o \
./source/ccMDP_GPU.o \
./source/ccMDP_GPU_modular.o \
./source/main.o 

CU_DEPS += \
./source/ccMDP_GPU.d \
./source/ccMDP_GPU_modular.d 

CPP_DEPS += \
./source/cMDPSquareManager.d \
./source/main.d 


# Each subdirectory must supply rules for building sources it contributes
source/%.o: ../source/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/home/benjha/cuda-workspace/GPUZen/MDP_Solver/header" -O3 -gencode arch=compute_35,code=sm_35  -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/home/benjha/cuda-workspace/GPUZen/MDP_Solver/header" -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

source/%.o: ../source/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/home/benjha/cuda-workspace/GPUZen/MDP_Solver/header" -O3 -gencode arch=compute_35,code=sm_35  -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/home/benjha/cuda-workspace/GPUZen/MDP_Solver/header" -O3 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


