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
 * Radix sort policy
 ******************************************************************************/

#pragma once

#include "../util/basic_utils.cuh"
#include "../util/io/modified_load.cuh"
#include "../util/io/modified_store.cuh"
#include "../util/ns_umbrella.cuh"

#include "../radix_sort/pass_policy.cuh"
#include "../radix_sort/upsweep/kernel_policy.cuh"
#include "../radix_sort/spine/kernel_policy.cuh"
#include "../radix_sort/downsweep/kernel_policy.cuh"
#include "../radix_sort/single/kernel_policy.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Dispatch policy
 ******************************************************************************/

/**
 * Alternative policies for how much dynamic smem should be allocated to each kernel
 */
enum DynamicSmemConfig
{
	DYNAMIC_SMEM_NONE,			// No dynamic smem for kernels
	DYNAMIC_SMEM_UNIFORM,		// Uniform: pad with dynamic smem so all kernels get the same total smem allocation
	DYNAMIC_SMEM_LCM,			// Least-common-multiple: pad with dynamic smem so kernel occupancy a multiple of the lowest occupancy
};


/**
 * Dispatch policy
 */
template <
	DynamicSmemConfig 	_DYNAMIC_SMEM_CONFIG,
	bool 				_UNIFORM_GRID_SIZE>
struct DispatchPolicy
{
	enum {
		UNIFORM_GRID_SIZE 			= _UNIFORM_GRID_SIZE,
	};

	static const DynamicSmemConfig 	DYNAMIC_SMEM_CONFIG = _DYNAMIC_SMEM_CONFIG;
};


/******************************************************************************
 * Pass policy
 ******************************************************************************/

/**
 * Pass policy
 */
template <
	typename 	_UpsweepPolicy,
	typename 	_SpinePolicy,
	typename 	_DownsweepPolicy,
	typename 	_SinglePolicy,
	typename 	_DispatchPolicy>
struct PassPolicy
{
	typedef _UpsweepPolicy			UpsweepPolicy;
	typedef _SpinePolicy 			SpinePolicy;
	typedef _DownsweepPolicy 		DownsweepPolicy;
	typedef _SinglePolicy 			SinglePolicy;
	typedef _DispatchPolicy 		DispatchPolicy;
};


/******************************************************************************
 * Tuned pass policy specializations
 ******************************************************************************/

/**
 * Problem size enumerations
 */
enum ProblemSize
{
	LARGE_PROBLEM,		// > 32K elements
	SMALL_PROBLEM		// <= 32K elements
};


/**
 * Preferred radix digit bits policy
 */
template <int TUNE_ARCH>
struct PreferredDigitBits
{
	enum {
		PREFERRED_BITS = 5,		// All architectures currently prefer 5-bit passes
	};
};


/**
 * Tuned pass policy specializations
 */
template <
	int 			TUNE_ARCH,
	typename 		ProblemInstance,
	ProblemSize 	PROBLEM_SIZE,
	int				RADIX_BITS>
struct TunedPassPolicy;


/**
 * SM20
 */
template <typename ProblemInstance, ProblemSize PROBLEM_SIZE, int RADIX_BITS>
struct TunedPassPolicy<200, ProblemInstance, PROBLEM_SIZE, RADIX_BITS>
{
	enum {
		TUNE_ARCH			= 200,
		KEYS_ONLY 			= util::Equals<typename ProblemInstance::ValueType, util::NullType>::VALUE,
		LARGE_DATA			= (sizeof(typename ProblemInstance::KeyType) > 4) || (sizeof(typename ProblemInstance::ValueType) > 4),
		EARLY_EXIT			= false,
	};

	// Dispatch policy
	typedef DispatchPolicy <
		DYNAMIC_SMEM_NONE, 						// UNIFORM_SMEM_ALLOCATION
		true> 									// UNIFORM_GRID_SIZE
			DispatchPolicy;

	// Upsweep kernel policy
	typedef upsweep::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		8,										// MIN_CTA_OCCUPANCY
		7,										// LOG_CTA_THREADS
		(!LARGE_DATA ? 2 : 1),					// LOG_LOAD_VEC_SIZE
		1,										// LOG_LOADS_PER_TILE
		b40c::util::io::ld::NONE,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte,			// SMEM_CONFIG
		EARLY_EXIT>								// EARLY_EXIT
			UpsweepPolicy;

	// Spine-scan kernel policy
	typedef spine::KernelPolicy<
		8,										// LOG_CTA_THREADS
		2,										// LOG_LOAD_VEC_SIZE
		2,										// LOG_LOADS_PER_TILE
		b40c::util::io::ld::NONE,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			SpinePolicy;

	// Downsweep kernel policy
	typedef downsweep::KernelPolicy<
		RADIX_BITS,														// RADIX_BITS
		(KEYS_ONLY && !LARGE_DATA ? 4 : 2), 							// MIN_CTA_OCCUPANCY
		(KEYS_ONLY && !LARGE_DATA ? 7 : 8),								// LOG_CTA_THREADS
		(((PROBLEM_SIZE != SMALL_PROBLEM) && !LARGE_DATA) ? 4 : 3),		// LOG_THREAD_ELEMENTS
		b40c::util::io::ld::tex,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		downsweep::SCATTER_TWO_PHASE,			// SCATTER_STRATEGY
		cudaSharedMemBankSizeFourByte,			// SMEM_CONFIG
		EARLY_EXIT>								// EARLY_EXIT
			DownsweepPolicy;

	// Single kernel policy
	typedef single::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		256,									// CTA_THREADS
		((KEYS_ONLY) ? 17 : 9), 				// THREAD_ELEMENTS
		b40c::util::io::ld::tex,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			SinglePolicy;
};


/**
 * SM13
 */
template <typename ProblemInstance, ProblemSize PROBLEM_SIZE, int RADIX_BITS>
struct TunedPassPolicy<130, ProblemInstance, PROBLEM_SIZE, RADIX_BITS>
{
	enum {
		TUNE_ARCH			= 130,
		KEYS_ONLY 			= util::Equals<typename ProblemInstance::ValueType, util::NullType>::VALUE,
		LARGE_DATA			= (sizeof(typename ProblemInstance::KeyType) > 4) || (sizeof(typename ProblemInstance::ValueType) > 4),
		EARLY_EXIT			= false,
	};

	// Dispatch policy
	typedef DispatchPolicy <
		DYNAMIC_SMEM_UNIFORM, 					// UNIFORM_SMEM_ALLOCATION
		true> 									// UNIFORM_GRID_SIZE
			DispatchPolicy;

	// Upsweep kernel policy
	typedef upsweep::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		(RADIX_BITS > 4 ? 3 : 6),				// MIN_CTA_OCCUPANCY
		7,										// LOG_CTA_THREADS
		0,										// LOG_LOAD_VEC_SIZE
		(!LARGE_DATA ? 2 : 1),					// LOG_LOADS_PER_TILE
		b40c::util::io::ld::NONE,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte,			// SMEM_CONFIG
		EARLY_EXIT>								// EARLY_EXIT
			UpsweepPolicy;

	// Spine-scan kernel policy
	typedef spine::KernelPolicy<
		8,										// LOG_CTA_THREADS
		2,										// LOG_LOAD_VEC_SIZE
		0,										// LOG_LOADS_PER_TILE
		b40c::util::io::ld::NONE,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			SpinePolicy;

	// Downsweep kernel policy
	typedef downsweep::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		(KEYS_ONLY ? 3 : 2),					// MIN_CTA_OCCUPANCY
		6,										// LOG_CTA_THREADS
		(!LARGE_DATA ? 4 : 3),					// LOG_THREAD_ELEMENTS
		b40c::util::io::ld::tex,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		downsweep::SCATTER_TWO_PHASE,			// SCATTER_STRATEGY
		cudaSharedMemBankSizeFourByte,			// SMEM_CONFIG
		EARLY_EXIT>								// EARLY_EXIT
			DownsweepPolicy;

	// Single kernel policy
	typedef single::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		64,										// CTA_THREADS
		((KEYS_ONLY) ? 17 : 9), 				// THREAD_ELEMENTS
		b40c::util::io::ld::tex,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			SinglePolicy;

};


/**
 * SM10
 */
template <typename ProblemInstance, ProblemSize PROBLEM_SIZE, int RADIX_BITS>
struct TunedPassPolicy<100, ProblemInstance, PROBLEM_SIZE, RADIX_BITS>
{
	enum {
		TUNE_ARCH			= 100,
		KEYS_ONLY 			= util::Equals<typename ProblemInstance::ValueType, util::NullType>::VALUE,
		LARGE_DATA			= (sizeof(typename ProblemInstance::KeyType) > 4) || (sizeof(typename ProblemInstance::ValueType) > 4),
		EARLY_EXIT			= false,
	};

	// Dispatch policy
	typedef DispatchPolicy <
		DYNAMIC_SMEM_LCM, 						// UNIFORM_SMEM_ALLOCATION
		true> 									// UNIFORM_GRID_SIZE
			DispatchPolicy;

	// Upsweep kernel policy
	typedef upsweep::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		2,										// MIN_CTA_OCCUPANCY
		7,										// LOG_CTA_THREADS
		0,										// LOG_LOAD_VEC_SIZE
		0,										// LOG_LOADS_PER_TILE
		b40c::util::io::ld::NONE,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte,			// SMEM_CONFIG
		EARLY_EXIT>								// EARLY_EXIT
			UpsweepPolicy;

	// Spine-scan kernel policy
	typedef spine::KernelPolicy<
		8,										// LOG_CTA_THREADS
		2,										// LOG_LOAD_VEC_SIZE
		0,										// LOG_LOADS_PER_TILE
		b40c::util::io::ld::NONE,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			SpinePolicy;

	// Downsweep kernel policy
	typedef downsweep::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		2,										// MIN_CTA_OCCUPANCY
		6,										// LOG_CTA_THREADS
		(!LARGE_DATA ? 4 : 3),					// LOG_THREAD_ELEMENTS
		b40c::util::io::ld::tex,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		downsweep::SCATTER_WARP_TWO_PHASE,		// SCATTER_STRATEGY
		cudaSharedMemBankSizeFourByte,			// SMEM_CONFIG
		EARLY_EXIT>								// EARLY_EXIT
			DownsweepPolicy;

	// Single kernel policy
	typedef single::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		64,										// CTA_THREADS
		((KEYS_ONLY) ? 17 : 9), 				// THREAD_ELEMENTS
		b40c::util::io::ld::tex,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			SinglePolicy;

};



}// namespace radix_sort
}// namespace b40c
B40C_NS_POSTFIX
