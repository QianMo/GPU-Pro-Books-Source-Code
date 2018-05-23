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
 * CTA collective abstraction for sorting keys (unsigned bits) by radix digit.
 ******************************************************************************/

#pragma once

#include "../radix_sort/sort_utils.cuh"

#include "../util/basic_utils.cuh"
#include "../util/reduction/serial_reduce.cuh"
#include "../util/scan/serial_scan.cuh"
#include "../util/ns_umbrella.cuh"


B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {


/**
 * CTA collective abstraction for sorting keys (unsigned bits) by radix digit.
 *
 * Keys must be in a form suitable for radix ranking (i.e., unsigned bits).
 */
template <
	typename				UnsignedBits,
	int 					CTA_THREADS,
	int						KEYS_PER_THREAD,
	int 					RADIX_BITS,
	typename 				ValueType = util::NullType,
	cudaSharedMemConfig 	SMEM_CONFIG = cudaSharedMemBankSizeFourByte>	// Shared memory bank size
class CtaRadixSort
{
private:

	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	enum
	{
		TILE_ELEMENTS				= CTA_THREADS * KEYS_PER_THREAD,

		LOG_MEM_BANKS				= CUB_LOG_MEM_BANKS(__CUB_CUDA_ARCH__),
		MEM_BANKS					= 1 << LOG_MEM_BANKS,

		// Insert padding if the number of keys per thread is a power of two
		PADDING  					= ((KEYS_PER_THREAD & (KEYS_PER_THREAD - 1)) == 0),

		PADDING_ELEMENTS			= (PADDING) ? (TILE_ELEMENTS >> LOG_MEM_BANKS) : 0,
	};

	// CtaRadixRank utility type
	typedef CtaRadixRank<
		CTA_THREADS,
		RADIX_BITS,
		SMEM_CONFIG> CtaRadixRank;

public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		union
		{
			typename CtaRadixRank::SmemStorage		ranking_storage;
			struct
			{
				UnsignedBits						key_exchange[TILE_ELEMENTS + PADDING_ELEMENTS];
				ValueType 							value_exchange[TILE_ELEMENTS + PADDING_ELEMENTS];
			};
		};
	};

private:

	//---------------------------------------------------------------------
	// Template iteration
	//---------------------------------------------------------------------

	template <typename T>
	static __device__ __forceinline__ void GatherThreadStride(
		T items[KEYS_PER_THREAD],
		T *buffer)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int shared_offset = (threadIdx.x * KEYS_PER_THREAD) + KEY;
			if (PADDING) shared_offset = util::SHR_ADD(shared_offset, LOG_MEM_BANKS, shared_offset);
			items[KEY] = buffer[shared_offset];
		}
	}

	template <typename T>
	static __device__ __forceinline__ void GatherCtaStride(
		T items[KEYS_PER_THREAD],
		T *buffer)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int shared_offset = (KEY * CTA_THREADS) + threadIdx.x;
			if (PADDING) shared_offset = util::SHR_ADD(shared_offset, LOG_MEM_BANKS, shared_offset);
			items[KEY] = buffer[shared_offset];
		}
	}

	template <typename T>
	static __device__ __forceinline__ void Scatter(
		unsigned int 	ranks[KEYS_PER_THREAD],
		T 				items[KEYS_PER_THREAD],
		T 				*buffer)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int shared_offset = ranks[KEY];
			if (PADDING) shared_offset = util::SHR_ADD(shared_offset, LOG_MEM_BANKS, shared_offset);

			buffer[shared_offset] = items[KEY];
		}
	}


public:


	//---------------------------------------------------------------------
	// Keys-only interface
	//---------------------------------------------------------------------

	/**
	 * Keys-only sorting
	 */
	static __device__ __forceinline__ void SortThreadToCtaStride(
		SmemStorage		&smem_storage,									// Shared memory storage
		UnsignedBits 	(&keys)[KEYS_PER_THREAD],
		unsigned int 	current_bit = 0,								// The least-significant bit needed for key comparison
		unsigned int	bits_remaining = sizeof(UnsignedBits) * 8)		// The number of bits needed for key comparison
	{
		// Radix sorting passes

		while (true)
		{
			unsigned int ranks[KEYS_PER_THREAD];

			// Rank the keys within the CTA
			CtaRadixRank::RankKeys(
				smem_storage.ranking_storage,
				keys,
				ranks,
				current_bit);

			__syncthreads();

			// Scatter keys to shared memory
			Scatter(ranks, keys, smem_storage.key_exchange);

			__syncthreads();

			current_bit += RADIX_BITS;
			if (current_bit >= bits_remaining) break;

			// Gather keys from shared memory (thread-stride)
			GatherThreadStride(keys, smem_storage.key_exchange);

			__syncthreads();
		}

		// Gather keys from shared memory (CTA-stride)
		GatherCtaStride(keys, smem_storage.key_exchange);
	}


	//---------------------------------------------------------------------
	// Keys-value interface
	//---------------------------------------------------------------------

	/**
	 * Keys-value sorting
	 */
	static __device__ __forceinline__ void SortThreadToCtaStride(
		SmemStorage		&smem_storage,									// Shared memory storage
		UnsignedBits 	(&keys)[KEYS_PER_THREAD],
		ValueType	 	(&values)[KEYS_PER_THREAD],
		unsigned int 	current_bit = 0,								// The least-significant bit needed for key comparison
		unsigned int	bits_remaining = sizeof(UnsignedBits) * 8)		// The number of bits needed for key comparison
	{
		// Radix sorting passes

		while (true)
		{
			unsigned int ranks[KEYS_PER_THREAD];

			// Rank the keys within the CTA
			CtaRadixRank::RankKeys(
				smem_storage.ranking_storage,
				keys,
				ranks,
				current_bit);

			__syncthreads();

			// Scatter keys and values to shared memory
			Scatter(ranks, keys, smem_storage.key_exchange);
			Scatter(ranks, values, smem_storage.value_exchange);

			__syncthreads();

			current_bit += RADIX_BITS;
			if (current_bit >= bits_remaining) break;

			// Gather keys and values from shared memory (thread-stride)
			GatherThreadStride(keys, smem_storage.key_exchange);
			GatherThreadStride(values, smem_storage.value_exchange);

			__syncthreads();
		}

		// Gather keys and values from shared memory (CTA-stride)
		GatherCtaStride(keys, smem_storage.key_exchange);
		GatherCtaStride(values, smem_storage.value_exchange);
	}
};




} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
