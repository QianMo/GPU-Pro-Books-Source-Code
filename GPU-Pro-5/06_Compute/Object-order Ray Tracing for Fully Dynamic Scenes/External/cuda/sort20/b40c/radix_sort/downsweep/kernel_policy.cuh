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
 * Configuration policy for radix sort downsweep scan kernel
 ******************************************************************************/

#pragma once

#include "../../util/io/modified_load.cuh"
#include "../../util/io/modified_store.cuh"
#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace downsweep {

/**
 * Types of scattering strategies
 */
enum ScatterStrategy
{
	SCATTER_DIRECT = 0,			// Scatter directly from registers to global bins
	SCATTER_TWO_PHASE,			// First scatter from registers into shared memory bins, then into global bins
	SCATTER_WARP_TWO_PHASE,		// Similar to SCATTER_TWO_PHASE, but with the additional constraint that each warp only perform segment-aligned global writes
};


/**
 * Downsweep tuning policy.
 */
template <
	int 							_RADIX_BITS,			// The number of radix bits, i.e., log2(bins)
	int 							_MIN_CTA_OCCUPANCY,		// The minimum CTA occupancy requested for this kernel per SM
	int 							_LOG_CTA_THREADS,		// The number of threads per CTA
	int 							_LOG_THREAD_ELEMENTS,	// The number of consecutive keys to process per thread per tile
	util::io::ld::CacheModifier	 	_LOAD_MODIFIER,			// Load cache-modifier
	util::io::st::CacheModifier 	_STORE_MODIFIER,		// Store cache-modifier
	ScatterStrategy 				_SCATTER_STRATEGY,		// Scattering strategy
	cudaSharedMemConfig				_SMEM_CONFIG,			// Shared memory bank size
	bool						 	_EARLY_EXIT>			// Whether or not to short-circuit passes if the upsweep determines homogoneous digits in the current digit place
struct KernelPolicy
{
	enum {
		RADIX_BITS					= _RADIX_BITS,
		MIN_CTA_OCCUPANCY  			= _MIN_CTA_OCCUPANCY,
		LOG_CTA_THREADS 			= _LOG_CTA_THREADS,
		LOG_THREAD_ELEMENTS 		= _LOG_THREAD_ELEMENTS,
		EARLY_EXIT					= _EARLY_EXIT,

		CTA_THREADS					= 1 << LOG_CTA_THREADS,
		LOG_TILE_ELEMENTS			= LOG_CTA_THREADS + LOG_THREAD_ELEMENTS,
	};

	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= _LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= _STORE_MODIFIER;
	static const cudaSharedMemConfig			SMEM_CONFIG			= _SMEM_CONFIG;
	static const ScatterStrategy 				SCATTER_STRATEGY 	= _SCATTER_STRATEGY;
};



} // namespace downsweep
} // namespace partition
} // namespace b40c
B40C_NS_POSTFIX
