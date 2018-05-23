/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 ******************************************************************************/

/******************************************************************************
 * CUDA Properties
 ******************************************************************************/

#pragma once

#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {

/******************************************************************************
 * Macros for guiding compilation paths
 ******************************************************************************/

/**
 * CUDA architecture of the current compilation path
 */
#ifndef __CUDA_ARCH__
	#define __CUB_CUDA_ARCH__ 0						// Host path
#else
	#define __CUB_CUDA_ARCH__ __CUDA_ARCH__			// Device path
#endif



/******************************************************************************
 * Device properties by SM architectural version
 ******************************************************************************/

// Invalid CUDA device ordinal
#define CUB_INVALID_DEVICE				(-1)

// Threads per warp. 
#define CUB_LOG_WARP_THREADS(arch)		(5)			// 32 threads in a warp 
#define CUB_WARP_THREADS(arch)			(1 << CUB_LOG_WARP_THREADS(arch))

// SM memory bank stride (in bytes)
#define CUB_LOG_BANK_STRIDE_BYTES(arch)	(2)		// 4 byte words
#define CUB_BANK_STRIDE_BYTES(arch)		(1 << CUB_LOG_BANK_STRIDE_BYTES)

// Memory banks per SM
#define CUB_SM20_LOG_MEM_BANKS()		(5)			// 32 banks on SM2.0+
#define CUB_SM10_LOG_MEM_BANKS()		(4)			// 16 banks on SM1.0-SM1.3
#define CUB_LOG_MEM_BANKS(arch)		((arch >= 200) ? CUB_SM20_LOG_MEM_BANKS() : 	\
														 CUB_SM10_LOG_MEM_BANKS())		

// Physical shared memory per SM (bytes)
#define CUB_SM20_SMEM_BYTES()			(49152)		// 48KB on SM2.0+
#define CUB_SM10_SMEM_BYTES()			(16384)		// 32KB on SM1.0-SM1.3
#define CUB_SMEM_BYTES(arch)			((arch >= 200) ? CUB_SM20_SMEM_BYTES() : 	\
														 CUB_SM10_SMEM_BYTES())		

// Physical threads per SM
#define CUB_SM30_SM_THREADS()			(2048)		// 2048 threads on SM3.0+
#define CUB_SM20_SM_THREADS()			(1536)		// 1536 threads on SM2.0
#define CUB_SM12_SM_THREADS()			(1024)		// 1024 threads on SM1.2-SM1.3
#define CUB_SM10_SM_THREADS()			(768)		// 768 threads on SM1.0-SM1.1
#define CUB_SM_THREADS(arch)			((arch >= 300) ? CUB_SM30_SM_THREADS() : 	\
										 (arch >= 200) ? CUB_SM20_SM_THREADS() : 	\
										 (arch >= 130) ? CUB_SM12_SM_THREADS() : 	\
												 	 	 CUB_SM10_SM_THREADS())

// Physical threads per CTA
#define CUB_SM20_LOG_CTA_THREADS()		(10)		// 1024 threads on SM2.0+
#define CUB_SM10_LOG_CTA_THREADS()		(9)			// 512 threads on SM1.0-SM1.3
#define CUB_LOG_CTA_THREADS(arch)		((arch >= 200) ? CUB_SM20_LOG_CTA_THREADS() : 	\
												 	 	 CUB_SM10_LOG_CTA_THREADS())

// Max CTAs per SM
#define CUB_SM30_SM_CTAS()				(16)	// 16 CTAs on SM3.0+
#define CUB_SM20_SM_CTAS()				(8)		// 8 CTAs on SM2.0+
#define CUB_SM12_SM_CTAS()				(8)		// 8 CTAs on SM1.2-SM1.3
#define CUB_SM10_SM_CTAS()				(8)		// 8 CTAs on SM1.0-SM1.1
#define CUB_SM_CTAS(arch)				((arch >= 300) ? CUB_SM30_SM_CTAS() : 	\
										 (arch >= 200) ? CUB_SM20_SM_CTAS() : 	\
										 (arch >= 130) ? CUB_SM12_SM_CTAS() : 	\
												 	 	 CUB_SM10_SM_CTAS())

// Max registers per SM
#define CUB_SM30_SM_REGISTERS()		(65536)		// 65536 registers on SM3.0+
#define CUB_SM20_SM_REGISTERS()		(32768)		// 32768 registers on SM2.0+
#define CUB_SM12_SM_REGISTERS()		(16384)		// 16384 registers on SM1.2-SM1.3
#define CUB_SM10_SM_REGISTERS()		(8192)		// 8192 registers on SM1.0-SM1.1
#define CUB_SM_REGISTERS(arch)			((arch >= 300) ? CUB_SM30_SM_REGISTERS() : 	\
										 (arch >= 200) ? CUB_SM20_SM_REGISTERS() : 	\
										 (arch >= 130) ? CUB_SM12_SM_REGISTERS() : 	\
												 	 	 CUB_SM10_SM_REGISTERS())

/******************************************************************************
 * Inlined PTX helper macros
 ******************************************************************************/


// Register modifier for pointer-types (for inlining PTX assembly)
#if defined(_WIN64) || defined(__LP64__)
	#define __CUB_LP64__ 1
	// 64-bit register modifier for inlined asm
	#define _CUB_ASM_PTR_ "l"
#else
	#define __CUB_LP64__ 0
	// 32-bit register modifier for inlined asm
	#define _CUB_ASM_PTR_ "r"
#endif



/******************************************************************************
 * CUDA/GPU inspection utilities
 ******************************************************************************/

/**
 * Empty Kernel
 */
template <typename T>
__global__ void FlushKernel(void) { }


/**
 * Class encapsulating device properties for dynamic host-side inspection
 */
class CudaProperties 
{
public:
	
	// Information about our target device
	cudaDeviceProp 		device_props;
	int 				device_sm_version;
	
	// Information about our kernel assembly
	int 				kernel_ptx_version;
	
public:
	
	/**
	 * Constructor
	 */
	CudaProperties() 
	{
		// Get current device properties 
		int current_device;
		cudaGetDevice(&current_device);
		cudaGetDeviceProperties(&device_props, current_device);
		device_sm_version = device_props.major * 100 + device_props.minor * 10;
	
		// Get SM version of compiled kernel assemblies
		cudaFuncAttributes flush_kernel_attrs;
		cudaFuncGetAttributes(&flush_kernel_attrs, FlushKernel<void>);
		kernel_ptx_version = flush_kernel_attrs.ptxVersion * 10;
	}

	/**
	 * Constructor
	 */
	CudaProperties(int gpu)
	{
		// Get current device properties
		cudaGetDeviceProperties(&device_props, gpu);
		device_sm_version = device_props.major * 100 + device_props.minor * 10;

		// Get SM version of compiled kernel assemblies
		cudaFuncAttributes flush_kernel_attrs;
		cudaFuncGetAttributes(&flush_kernel_attrs, FlushKernel<void>);
		kernel_ptx_version = flush_kernel_attrs.ptxVersion * 10;
	}
};



} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
