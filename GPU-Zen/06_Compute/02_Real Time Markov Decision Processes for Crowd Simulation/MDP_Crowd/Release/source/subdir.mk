################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../source/ccMDP_GPU.cu \
../source/ccMDP_GPU_modular.cu \
../source/ccMDP_LCA_paths.cu \
../source/ccMDP_hexagon_paths.cu \
../source/ccMDP_square_paths.cu 

CPP_SRCS += \
../source/cAnimation.cpp \
../source/cAxes.cpp \
../source/cCamera.cpp \
../source/cCharacterGroup.cpp \
../source/cCharacterModel.cpp \
../source/cCrowd.cpp \
../source/cCrowdGroup.cpp \
../source/cCrowdManager.cpp \
../source/cCrowdParser.cpp \
../source/cFboManager.cpp \
../source/cFrameBuffer.cpp \
../source/cFrustum.cpp \
../source/cGlErrorManager.cpp \
../source/cGlslManager.cpp \
../source/cLogManager.cpp \
../source/cMDPCudaPathManager.cpp \
../source/cMDPHexagonManager.cpp \
../source/cMDPSquareManager.cpp \
../source/cModel3D.cpp \
../source/cModelProps.cpp \
../source/cObstacleManager.cpp \
../source/cProjectionManager.cpp \
../source/cRenderBuffer.cpp \
../source/cScenario.cpp \
../source/cScreenText.cpp \
../source/cShaderObject.cpp \
../source/cSkyboxManager.cpp \
../source/cStaticLod.cpp \
../source/cStringUtils.cpp \
../source/cTexture.cpp \
../source/cTexture3D.cpp \
../source/cTextureManager.cpp \
../source/cTimer.cpp \
../source/cVboManager.cpp \
../source/cXmlParser.cpp \
../source/main.cpp \
../source/tinyxml.cpp \
../source/tinyxmlerror.cpp \
../source/tinyxmlparser.cpp 

OBJS += \
./source/cAnimation.o \
./source/cAxes.o \
./source/cCamera.o \
./source/cCharacterGroup.o \
./source/cCharacterModel.o \
./source/cCrowd.o \
./source/cCrowdGroup.o \
./source/cCrowdManager.o \
./source/cCrowdParser.o \
./source/cFboManager.o \
./source/cFrameBuffer.o \
./source/cFrustum.o \
./source/cGlErrorManager.o \
./source/cGlslManager.o \
./source/cLogManager.o \
./source/cMDPCudaPathManager.o \
./source/cMDPHexagonManager.o \
./source/cMDPSquareManager.o \
./source/cModel3D.o \
./source/cModelProps.o \
./source/cObstacleManager.o \
./source/cProjectionManager.o \
./source/cRenderBuffer.o \
./source/cScenario.o \
./source/cScreenText.o \
./source/cShaderObject.o \
./source/cSkyboxManager.o \
./source/cStaticLod.o \
./source/cStringUtils.o \
./source/cTexture.o \
./source/cTexture3D.o \
./source/cTextureManager.o \
./source/cTimer.o \
./source/cVboManager.o \
./source/cXmlParser.o \
./source/ccMDP_GPU.o \
./source/ccMDP_GPU_modular.o \
./source/ccMDP_LCA_paths.o \
./source/ccMDP_hexagon_paths.o \
./source/ccMDP_square_paths.o \
./source/main.o \
./source/tinyxml.o \
./source/tinyxmlerror.o \
./source/tinyxmlparser.o 

CU_DEPS += \
./source/ccMDP_GPU.d \
./source/ccMDP_GPU_modular.d \
./source/ccMDP_LCA_paths.d \
./source/ccMDP_hexagon_paths.d \
./source/ccMDP_square_paths.d 

CPP_DEPS += \
./source/cAnimation.d \
./source/cAxes.d \
./source/cCamera.d \
./source/cCharacterGroup.d \
./source/cCharacterModel.d \
./source/cCrowd.d \
./source/cCrowdGroup.d \
./source/cCrowdManager.d \
./source/cCrowdParser.d \
./source/cFboManager.d \
./source/cFrameBuffer.d \
./source/cFrustum.d \
./source/cGlErrorManager.d \
./source/cGlslManager.d \
./source/cLogManager.d \
./source/cMDPCudaPathManager.d \
./source/cMDPHexagonManager.d \
./source/cMDPSquareManager.d \
./source/cModel3D.d \
./source/cModelProps.d \
./source/cObstacleManager.d \
./source/cProjectionManager.d \
./source/cRenderBuffer.d \
./source/cScenario.d \
./source/cScreenText.d \
./source/cShaderObject.d \
./source/cSkyboxManager.d \
./source/cStaticLod.d \
./source/cStringUtils.d \
./source/cTexture.d \
./source/cTexture3D.d \
./source/cTextureManager.d \
./source/cTimer.d \
./source/cVboManager.d \
./source/cXmlParser.d \
./source/main.d \
./source/tinyxml.d \
./source/tinyxmlerror.d \
./source/tinyxmlparser.d 


# Each subdirectory must supply rules for building sources it contributes
source/%.o: ../source/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -DTIXML_USE_STL -I"/home/benjha/cuda-workspace/GPUZen/MDP_Crowd/header" -O3 -std=c++11 -gencode arch=compute_35,code=sm_35  -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -DTIXML_USE_STL -I"/home/benjha/cuda-workspace/GPUZen/MDP_Crowd/header" -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

source/%.o: ../source/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -DTIXML_USE_STL -I"/home/benjha/cuda-workspace/GPUZen/MDP_Crowd/header" -O3 -std=c++11 -gencode arch=compute_35,code=sm_35  -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -DTIXML_USE_STL -I"/home/benjha/cuda-workspace/GPUZen/MDP_Crowd/header" -O3 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


