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
 * CTA-processing functionality for scan kernels
 ******************************************************************************/

#pragma once

#include "../../util/cuda_properties.cuh"
#include "../../util/srts_grid.cuh"
#include "../../util/srts_details.cuh"
#include "../../util/operators.cuh"
#include "../../util/io/modified_load.cuh"
#include "../../util/io/modified_store.cuh"
#include "../../util/io/load_tile.cuh"
#include "../../util/io/store_tile.cuh"
#include "../../util/scan/cooperative_scan.cuh"
#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace spine {


template <
	typename KernelPolicy,
	typename T,
	typename SizeT>
struct Cta
{
	//---------------------------------------------------------------------
	// Constants and type definitions
	//---------------------------------------------------------------------

	enum {
		LOG_CTA_THREADS 					= KernelPolicy::LOG_CTA_THREADS,
		CTA_THREADS							= 1 << LOG_CTA_THREADS,

		LOG_LOAD_VEC_SIZE  				= KernelPolicy::LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 				= KernelPolicy::LOG_LOADS_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,

		LOG_WARP_THREADS 				= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS					= 1 << LOG_WARP_THREADS,

		LOG_WARPS						= LOG_CTA_THREADS - LOG_WARP_THREADS,
		WARPS							= 1 << LOG_WARPS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_CTA_THREADS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,
	};

	/**
	 * Raking grid type
	 */
	typedef util::RakingGrid<
		T,										// Partial type
		LOG_CTA_THREADS,							// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_WARP_THREADS,						// 1 warp of raking threads
		true>									// There are prefix dependences between lanes
			RakingGrid;

	/**
	 * Operational details type for raking grid type
	 */
	typedef util::RakingDetails<RakingGrid> RakingDetails;


	/**
	 * Shared memory storage type
	 */
	struct SmemStorage
	{
		T warpscan[2][WARP_THREADS];
		T raking_elements[RakingGrid::TOTAL_RAKING_ELEMENTS];		// Raking raking elements
	};


	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage &smem_storage;

	// Running partial accumulated by the CTA over its tile-processing
	// lifetime (managed in each raking thread)
	T carry;

	// Input and output device pointers
	T *d_in;
	T *d_out;

	// Operational details for raking scan grid
	RakingDetails raking_details;

	// Scan operator
	util::Sum<T> scan_op;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 		&smem_storage,
		T 					*d_in,
		T 					*d_out) :
			// Initializers
			raking_details(
				smem_storage.raking_elements,
				smem_storage.warpscan,
				0),
			smem_storage(smem_storage),
			d_in(d_in),
			d_out(d_out),
			carry(0)
	{}

	/**
	 * Process a single tile
	 */
	__device__ __forceinline__ void ProcessTile(SizeT cta_offset)
	{
		// Tile of scan elements
		T partials[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Load tile
		util::io::LoadTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			CTA_THREADS,
			KernelPolicy::LOAD_MODIFIER,
			false>::LoadValid(
				partials,
				d_in,
				cta_offset);

		// Scan tile with carry update in raking threads
		util::scan::CooperativeTileScan<
			LOAD_VEC_SIZE,
			true>::ScanTileWithCarry(
				raking_details,
				partials,
				carry,
				scan_op);

		// Store tile
		util::io::StoreTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			CTA_THREADS,
			KernelPolicy::STORE_MODIFIER,
			false>::Store(
				partials,
				d_out,
				cta_offset);
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		SizeT num_elements)
	{
		for (SizeT cta_offset = 0;
			cta_offset < num_elements;
			cta_offset += TILE_ELEMENTS)
		{
			// Process full tiles of tile_elements
			ProcessTile(cta_offset);
		}
	}

};


} // namespace spine
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
