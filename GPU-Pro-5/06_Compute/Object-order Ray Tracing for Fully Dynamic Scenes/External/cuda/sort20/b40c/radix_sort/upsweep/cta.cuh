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
 * CTA-processing functionality for radix sort upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include "../../radix_sort/sort_utils.cuh"
#include "../../util/basic_utils.cuh"
#include "../../util/device_intrinsics.cuh"
#include "../../util/io/load_tile.cuh"
#include "../../util/reduction/serial_reduce.cuh"
#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace upsweep {

/**
 * Radix sort upsweep reduction CTA
 */
template <
	typename KernelPolicy,
	typename SizeT,
	typename KeyType>
struct Cta
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	typedef typename KeyTraits<KeyType>::UnsignedBits 	UnsignedBits;

	// Integer type for digit counters (to be packed into words of PackedCounters)
	typedef unsigned char DigitCounter;

	// Integer type for packing DigitCounters into columns of shared memory banks
	typedef typename util::If<(KernelPolicy::SMEM_CONFIG == cudaSharedMemBankSizeEightByte),
		unsigned long long,
		unsigned int>::Type PackedCounter;

	enum {
		RADIX_BITS					= KernelPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,

		LOG_CTA_THREADS 			= KernelPolicy::LOG_CTA_THREADS,
		CTA_THREADS					= 1 << LOG_CTA_THREADS,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		LOG_WARPS					= LOG_CTA_THREADS - LOG_WARP_THREADS,
		WARPS						= 1 << LOG_WARPS,

		LOG_LOAD_VEC_SIZE  			= KernelPolicy::LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE				= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 			= KernelPolicy::LOG_LOADS_PER_TILE,
		LOADS_PER_TILE				= 1 << LOG_LOADS_PER_TILE,

		LOG_THREAD_ELEMENTS			= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		THREAD_ELEMENTS				= 1 << LOG_THREAD_ELEMENTS,

		LOG_TILE_ELEMENTS 			= LOG_THREAD_ELEMENTS + LOG_CTA_THREADS,
		TILE_ELEMENTS				= 1 << LOG_TILE_ELEMENTS,

		BYTES_PER_COUNTER			= sizeof(DigitCounter),
		LOG_BYTES_PER_COUNTER		= util::Log2<BYTES_PER_COUNTER>::VALUE,

		PACKING_RATIO				= sizeof(PackedCounter) / sizeof(DigitCounter),
		LOG_PACKING_RATIO			= util::Log2<PACKING_RATIO>::VALUE,

		LOG_COUNTER_LANES 			= CUB_MAX(0, RADIX_BITS - LOG_PACKING_RATIO),
		COUNTER_LANES 				= 1 << LOG_COUNTER_LANES,

		// To prevent counter overflow, we must periodically unpack and aggregate the
		// digit counters back into registers.  Each counter lane is assigned to a
		// warp for aggregation.

		LOG_LANES_PER_WARP			= CUB_MAX(0, LOG_COUNTER_LANES - LOG_WARPS),
		LANES_PER_WARP 				= 1 << LOG_LANES_PER_WARP,

		// Unroll tiles in batches without risk of counter overflow
		UNROLL_COUNT				= CUB_MIN(64, 255 / THREAD_ELEMENTS),
		UNROLLED_ELEMENTS 			= UNROLL_COUNT * TILE_ELEMENTS,
	};



	/**
	 * Shared storage for radix distribution sorting upsweep
	 */
	struct SmemStorage
	{
		union
		{
			DigitCounter 	digit_counters[COUNTER_LANES][CTA_THREADS][PACKING_RATIO];
			PackedCounter 	packed_counters[COUNTER_LANES][CTA_THREADS];
			SizeT 			digit_partials[RADIX_DIGITS][WARP_THREADS + 1];
		};
	};


	//---------------------------------------------------------------------
	// Thread fields
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage 		&smem_storage;

	// Thread-local counters for periodically aggregating composite-counter lanes
	SizeT 				local_counts[LANES_PER_WARP][PACKING_RATIO];

	// Input and output device pointers
	UnsignedBits		*d_in_keys;
	SizeT				*d_spine;

	// The least-significant bit position of the current digit to extract
	unsigned int 		current_bit;



	//---------------------------------------------------------------------
	// Helper structure for templated iteration
	//---------------------------------------------------------------------

	// Iterate
	template <int COUNT, int MAX>
	struct Iterate
	{
		enum {
			HALF = (MAX / 2),
		};

		// BucketKeys
		static __device__ __forceinline__ void BucketKeys(
			Cta &cta,
			UnsignedBits keys[THREAD_ELEMENTS])
		{
			cta.Bucket(keys[COUNT]);

			// Next
			Iterate<COUNT + 1, MAX>::BucketKeys(cta, keys);
		}

		// ProcessTiles
		static __device__ __forceinline__ void ProcessTiles(Cta &cta, SizeT cta_offset)
		{
			// Next
			Iterate<1, HALF>::ProcessTiles(cta, cta_offset);
			Iterate<1, MAX - HALF>::ProcessTiles(cta, cta_offset + (HALF * TILE_ELEMENTS));
		}
	};

	// Terminate
	template <int MAX>
	struct Iterate<MAX, MAX>
	{
		// BucketKeys
		static __device__ __forceinline__ void BucketKeys(Cta &cta, UnsignedBits keys[THREAD_ELEMENTS]) {}

		// ProcessTiles
		static __device__ __forceinline__ void ProcessTiles(Cta &cta, SizeT cta_offset)
		{
			cta.ProcessFullTile(cta_offset);
		}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		SizeT 			*d_spine,
		KeyType 		*d_in_keys,
		unsigned int 	current_bit) :
			smem_storage(smem_storage),
			d_in_keys(reinterpret_cast<UnsignedBits*>(d_in_keys)),
			d_spine(d_spine),
			current_bit(current_bit)
	{}


	/**
	 * Decode a key and increment corresponding smem digit counter
	 */
	__device__ __forceinline__ void Bucket(UnsignedBits key)
	{
		// Perform transform op
		UnsignedBits converted_key = KeyTraits<KeyType>::TwiddleIn(key);

		// Add in sub-counter offset
		UnsignedBits sub_counter = util::BFE(converted_key, current_bit, LOG_PACKING_RATIO);

		// Add in row offset
		UnsignedBits row_offset = util::BFE(converted_key, current_bit + LOG_PACKING_RATIO, LOG_COUNTER_LANES);

		// Increment counter
		smem_storage.digit_counters[row_offset][threadIdx.x][sub_counter]++;

	}


	/**
	 * Reset composite counters
	 */
	__device__ __forceinline__ void ResetDigitCounters()
	{
		#pragma unroll
		for (int LANE = 0; LANE < COUNTER_LANES; LANE++)
		{
			smem_storage.packed_counters[LANE][threadIdx.x] = 0;
		}
	}


	/**
	 * Reset the unpacked counters in each thread
	 */
	__device__ __forceinline__ void ResetUnpackedCounters()
	{
		#pragma unroll
		for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
		{
			#pragma unroll
			for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
			{
				local_counts[LANE][UNPACKED_COUNTER] = 0;
			}
		}
	}


	/**
	 * Extracts and aggregates the digit counters for each counter lane
	 * owned by this warp
	 */
	__device__ __forceinline__ void UnpackDigitCounts()
	{
		unsigned int warp_id = threadIdx.x >> LOG_WARP_THREADS;
		unsigned int warp_tid = threadIdx.x & (WARP_THREADS - 1);

		if (warp_id < COUNTER_LANES)
		{
			#pragma unroll
			for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
			{
				const int COUNTER_LANE = LANE * WARPS;

				#pragma unroll
				for (int PACKED_COUNTER = 0; PACKED_COUNTER < CTA_THREADS; PACKED_COUNTER += WARP_THREADS)
				{
					#pragma unroll
					for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
					{
						const int OFFSET = (((COUNTER_LANE * CTA_THREADS) + PACKED_COUNTER) * PACKING_RATIO) + UNPACKED_COUNTER;
						local_counts[LANE][UNPACKED_COUNTER] += smem_storage.digit_counters[warp_id][warp_tid][OFFSET];
					}
				}
			}
		}
	}


	/**
	 * Places unpacked counters into smem for final digit reduction
	 */
	__device__ __forceinline__ void ReduceUnpackedCounts()
	{
		unsigned int warp_id = threadIdx.x >> LOG_WARP_THREADS;
		unsigned int warp_tid = threadIdx.x & (WARP_THREADS - 1);

		// Place unpacked digit counters in shared memory
		if (warp_id < COUNTER_LANES)
		{
			#pragma unroll
			for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
			{
				const int COUNTER_LANE = LANE * WARPS;
				int digit_row = (COUNTER_LANE + warp_id) << LOG_PACKING_RATIO;

				#pragma unroll
				for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
				{
					smem_storage.digit_partials[digit_row + UNPACKED_COUNTER][warp_tid]
						  = local_counts[LANE][UNPACKED_COUNTER];
				}
			}
		}

		__syncthreads();

		// Rake-reduce and write out the bin_count reductions
		if (threadIdx.x < RADIX_DIGITS)
		{
			SizeT bin_count = util::reduction::SerialReduce<WARP_THREADS>::Invoke(
				smem_storage.digit_partials[threadIdx.x]);

			int spine_bin_offset = (gridDim.x * threadIdx.x) + blockIdx.x;

			util::io::ModifiedStore<KernelPolicy::STORE_MODIFIER>::St(
				bin_count,
				d_spine + spine_bin_offset);
		}
	}


	/**
	 * Processes a single, full tile
	 */
	__device__ __forceinline__ void ProcessFullTile(SizeT cta_offset)
	{
		// Tile of keys
		UnsignedBits keys[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Read tile of keys
		util::io::LoadTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			CTA_THREADS,
			KernelPolicy::LOAD_MODIFIER,
			false>::LoadValid(
				keys,
				d_in_keys,
				cta_offset);

		// Prevent bucketing from being hoisted (otherwise we don't get the desired outstanding loads)
		if (LOADS_PER_TILE > 1) __syncthreads();

		// Bucket tile of keys
		Iterate<0, THREAD_ELEMENTS>::BucketKeys(*this, (UnsignedBits*) keys);
	}


	/**
	 * Processes a single load (may have some threads masked off)
	 */
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		const SizeT &out_of_bounds)
	{
		// Process partial tile if necessary using single loads
		cta_offset += threadIdx.x;
		while (cta_offset < out_of_bounds)
		{
			// Load and bucket key
			UnsignedBits key = d_in_keys[cta_offset];
			Bucket(key);
			cta_offset += CTA_THREADS;
		}
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Reset digit counters in smem and unpacked counters in registers
		ResetDigitCounters();
		ResetUnpackedCounters();

		SizeT cta_offset = work_limits.offset;

		// Unroll batches of full tiles
		while (cta_offset + UNROLLED_ELEMENTS < work_limits.out_of_bounds)
		{
			Iterate<0, UNROLL_COUNT>::ProcessTiles(*this, cta_offset);
			cta_offset += UNROLLED_ELEMENTS;

			__syncthreads();

			// Aggregate back into local_count registers to prevent overflow
			UnpackDigitCounts();

			__syncthreads();

			// Reset composite counters in lanes
			ResetDigitCounters();
		}

		// Unroll single full tiles
		while (cta_offset < work_limits.guarded_offset)
		{
			ProcessFullTile(cta_offset);
			cta_offset += TILE_ELEMENTS;
		}

		// Process partial tile if necessary
		ProcessPartialTile(cta_offset, work_limits.out_of_bounds);

		__syncthreads();

		// Aggregate back into local_count registers
		UnpackDigitCounts();

		__syncthreads();

		// Final raking reduction of counts by bin, output to spine.
		ReduceUnpackedCounts();
	}
};



} // namespace upsweep
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
