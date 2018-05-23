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
 * CTA-processing functionality for radix sort downsweep scan kernels
 ******************************************************************************/

#pragma once

#include "../../util/basic_utils.cuh"
#include "../../util/cta_work_distribution.cuh"
#include "../../util/tex_vector.cuh"
#include "../../util/io/modified_load.cuh"
#include "../../util/io/modified_store.cuh"
#include "../../util/ns_umbrella.cuh"

#include "../../radix_sort/sort_utils.cuh"
#include "../../radix_sort/cta_radix_rank.cuh"
#include "../../radix_sort/tex_ref.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace downsweep {


/**
 * Partitioning downsweep scan CTA
 */
template <
	typename KernelPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
struct Cta
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	// Appropriate unsigned-bits representation of KeyType
	typedef typename KeyTraits<KeyType>::UnsignedBits UnsignedBits;

	static const UnsignedBits 					MIN_KEY 			= KeyTraits<KeyType>::MIN_KEY;
	static const UnsignedBits 					MAX_KEY 			= KeyTraits<KeyType>::MAX_KEY;
	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= KernelPolicy::LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= KernelPolicy::STORE_MODIFIER;
	static const ScatterStrategy 				SCATTER_STRATEGY 	= KernelPolicy::SCATTER_STRATEGY;

	enum {
		RADIX_BITS					= KernelPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,
		KEYS_ONLY 					= util::Equals<ValueType, util::NullType>::VALUE,

		LOG_CTA_THREADS 			= KernelPolicy::LOG_CTA_THREADS,
		CTA_THREADS					= 1 << LOG_CTA_THREADS,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		LOG_WARPS					= LOG_CTA_THREADS - LOG_WARP_THREADS,
		WARPS						= 1 << LOG_WARPS,

		LOG_KEYS_PER_THREAD 		= KernelPolicy::LOG_THREAD_ELEMENTS,
		KEYS_PER_THREAD				= 1 << LOG_KEYS_PER_THREAD,

		LOG_TILE_ELEMENTS			= LOG_CTA_THREADS + LOG_KEYS_PER_THREAD,
		TILE_ELEMENTS				= 1 << LOG_TILE_ELEMENTS,

		BYTES_PER_SIZET				= sizeof(SizeT),
		LOG_BYTES_PER_SIZET			= util::Log2<BYTES_PER_SIZET>::VALUE,

		LOG_MEM_BANKS				= CUB_LOG_MEM_BANKS(__CUB_CUDA_ARCH__),
		MEM_BANKS					= 1 << LOG_MEM_BANKS,

		// Whether or not to insert padding for exchanging keys. (Padding is
		// worse than bank conflicts on GPUs that need two-phase scattering)
		PADDED_EXCHANGE 			= (SCATTER_STRATEGY != SCATTER_WARP_TWO_PHASE),
		PADDING_ELEMENTS			= (PADDED_EXCHANGE) ? (TILE_ELEMENTS >> LOG_MEM_BANKS) : 0,

		DIGITS_PER_SCATTER_PASS 	= CTA_THREADS / MEM_BANKS,
		SCATTER_PASSES 				= RADIX_DIGITS / DIGITS_PER_SCATTER_PASS,

		LOG_STORE_TXN_THREADS 		= LOG_MEM_BANKS,
		STORE_TXN_THREADS 			= 1 << LOG_STORE_TXN_THREADS,

		ELEMENTS_PER_TEX			= Textures<KeyType, ValueType, KEYS_PER_THREAD>::ELEMENTS_PER_TEX,

		THREAD_TEX_LOADS	 		= KEYS_PER_THREAD / ELEMENTS_PER_TEX,

		TILE_TEX_LOADS				= CTA_THREADS * THREAD_TEX_LOADS,
	};

	// Texture types
	typedef Textures<KeyType, ValueType, KEYS_PER_THREAD> 					Textures;
	typedef typename Textures::KeyTexType 									KeyTexType;
	typedef typename Textures::ValueTexType 								ValueTexType;

	// CtaRadixRank utility type
	typedef CtaRadixRank<
		CTA_THREADS,
		RADIX_BITS,
		KernelPolicy::SMEM_CONFIG> CtaRadixRank;

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		SizeT							tex_offset;
		SizeT							tex_offset_limit;

		util::CtaWorkLimits<SizeT> 		work_limits;

		unsigned int 					digit_prefixes[RADIX_DIGITS + 1];

		union
		{
			SizeT 						digit_offsets[RADIX_DIGITS];
		};

		union
		{
			typename CtaRadixRank::SmemStorage	ranking_storage;
			UnsignedBits						key_exchange[TILE_ELEMENTS + PADDING_ELEMENTS];
			ValueType 							value_exchange[TILE_ELEMENTS + PADDING_ELEMENTS];
		};
	};


	//---------------------------------------------------------------------
	// Thread fields
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage 				&smem_storage;

	// Input and output device pointers
	UnsignedBits 				*d_in_keys;
	UnsignedBits				*d_out_keys;
	ValueType 					*d_in_values;
	ValueType 					*d_out_values;

	// The global scatter base offset for each digit (valid in the first RADIX_DIGITS threads)
	SizeT 						my_digit_offset;

	// The least-significant bit position of the current digit to extract
	unsigned int 				current_bit;


	//---------------------------------------------------------------------
	// Helper structure for templated iteration.  (NVCC currently won't
	// unroll loops with "unexpected control flow".)
	//---------------------------------------------------------------------

	/**
	 * Iterate
	 */
	template <int COUNT, int MAX>
	struct Iterate
	{
		/**
		 * Scatter items to global memory
		 */
		template <bool FULL_TILE, typename T>
		static __device__ __forceinline__ void ScatterGlobal(
			T 				items[KEYS_PER_THREAD],
			SizeT			digit_offsets[KEYS_PER_THREAD],
			T 				*d_out,
			SizeT 			guarded_elements)
		{
			// Scatter if not out-of-bounds
			int tile_element = threadIdx.x + (COUNT * CTA_THREADS);
			T* scatter = d_out + threadIdx.x + (COUNT * CTA_THREADS) + digit_offsets[COUNT];

			if (FULL_TILE || (tile_element < guarded_elements))
			{
				util::io::ModifiedStore<STORE_MODIFIER>::St(items[COUNT], scatter);
			}

			// Iterate next element
			Iterate<COUNT + 1, MAX>::template ScatterGlobal<FULL_TILE>(
				items, digit_offsets, d_out, guarded_elements);
		}


		/**
		 * Scatter items to global memory
		 */
		template <bool FULL_TILE, typename T>
		static __device__ __forceinline__ void ScatterGlobal(
			T 				items[KEYS_PER_THREAD],
			unsigned int 	ranks[KEYS_PER_THREAD],
			SizeT			digit_offsets[KEYS_PER_THREAD],
			T 				*d_out,
			SizeT 			guarded_elements)
		{
			// Scatter if not out-of-bounds
			T* scatter = d_out + ranks[COUNT] + digit_offsets[COUNT];

			if (FULL_TILE || (ranks[COUNT] < guarded_elements))
			{
				util::io::ModifiedStore<STORE_MODIFIER>::St(items[COUNT], scatter);
			}

			// Iterate next element
			Iterate<COUNT + 1, MAX>::template ScatterGlobal<FULL_TILE>(
				items, ranks, digit_offsets, d_out, guarded_elements);
		}


		/**
		 * Warp based scattering that does not cross alignment boundaries, e.g., for SM1.0-1.1
		 * coalescing rules
		 */
		template <typename T>
		static __device__ __forceinline__ void AlignedScatterPass(
			SmemStorage 	&smem_storage,
			T 				*buffer,
			T 				*d_out,
			SizeT 			valid_elements)
		{
			int store_txn_idx 		= threadIdx.x & (STORE_TXN_THREADS - 1);
			int store_txn_digit 	= threadIdx.x >> LOG_STORE_TXN_THREADS;
			int my_digit 			= (COUNT * DIGITS_PER_SCATTER_PASS) + store_txn_digit;

			if (my_digit < RADIX_DIGITS)
			{
				int my_exclusive_scan 	= smem_storage.digit_prefixes[my_digit];
				int my_inclusive_scan 	= smem_storage.digit_prefixes[my_digit + 1];
				int my_carry 			= smem_storage.digit_offsets[my_digit] + my_exclusive_scan;
				int my_aligned_offset 	= store_txn_idx - (my_carry & (STORE_TXN_THREADS - 1));

				int gather_offset;
				while ((gather_offset = my_aligned_offset + my_exclusive_scan) < my_inclusive_scan)
				{
					if ((my_aligned_offset >= 0) && (gather_offset < valid_elements))
					{
						int padded_gather_offset = (PADDED_EXCHANGE) ?
							gather_offset = util::SHR_ADD(gather_offset, LOG_MEM_BANKS, gather_offset) :
							gather_offset;

						T datum = buffer[padded_gather_offset];
						d_out[my_carry + my_aligned_offset] = datum;
					}
					my_aligned_offset += STORE_TXN_THREADS;
				}
			}

			// Next scatter pass
			Iterate<COUNT + 1, MAX>::AlignedScatterPass(smem_storage, buffer, d_out, valid_elements);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int MAX>
	struct Iterate<MAX, MAX>
	{
		// ScatterGlobal
		template <bool FULL_TILE, typename T>
		static __device__ __forceinline__ void ScatterGlobal(T[KEYS_PER_THREAD], SizeT[KEYS_PER_THREAD], T*, SizeT) {}

		// ScatterGlobal
		template <bool FULL_TILE, typename T>
		static __device__ __forceinline__ void ScatterGlobal(T[KEYS_PER_THREAD], unsigned int[KEYS_PER_THREAD], SizeT[KEYS_PER_THREAD], T*, SizeT) {}

		// AlignedScatterPass
		template <typename T>
		static __device__ __forceinline__ void AlignedScatterPass(SmemStorage&, T*, T*, SizeT) {}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		KeyType 		*d_out_keys,
		ValueType 		*d_in_values,
		ValueType 		*d_out_values,
		SizeT 			*d_spine,
		unsigned int 	current_bit) :
			smem_storage(smem_storage),
			d_in_keys(reinterpret_cast<UnsignedBits*>(d_in_keys)),
			d_out_keys(reinterpret_cast<UnsignedBits*>(d_out_keys)),
			d_in_values(d_in_values),
			d_out_values(d_out_values),
			current_bit(current_bit)
	{
		if ((CTA_THREADS == RADIX_DIGITS) || (threadIdx.x < RADIX_DIGITS))
		{
			// Read digit scatter base (in parallel)
			int spine_digit_offset = (gridDim.x * threadIdx.x) + blockIdx.x;
			my_digit_offset = d_spine[spine_digit_offset];
		}
	}


	/**
	 * Perform a bit-wise twiddling transformation on keys
	 */
	template <UnsignedBits TwiddleOp(UnsignedBits)>
	__device__ __forceinline__ void TwiddleKeys(
		UnsignedBits keys[KEYS_PER_THREAD],
		UnsignedBits twiddled_keys[KEYS_PER_THREAD])		// out parameter
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			twiddled_keys[KEY] = TwiddleOp(keys[KEY]);
		}
	}


	/**
	 * Scatter ranked items to shared memory buffer
	 */
	template <typename T>
	__device__ __forceinline__ void ScatterRanked(
		unsigned int 	ranks[KEYS_PER_THREAD],
		T 				items[KEYS_PER_THREAD],
		T 				*buffer)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int offset = ranks[KEY];

			if (PADDED_EXCHANGE)
			{
				// Workaround for (CUAD4.2+NVCC+abi+m64) bug when sorting 16-bit key-value pairs
				offset = (sizeof(ValueType) == 2) ?
					(offset >> LOG_MEM_BANKS) + offset :
					util::SHR_ADD(offset, LOG_MEM_BANKS, offset);
			}

			buffer[offset] = items[KEY];
		}
	}


	/**
	 * Gather items from shared memory buffer
	 */
	template <typename T>
	__device__ __forceinline__ void GatherShared(
		T items[KEYS_PER_THREAD],
		T *buffer)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int gather_offset = (PADDED_EXCHANGE) ?
				(util::SHR_ADD(threadIdx.x, LOG_MEM_BANKS, threadIdx.x) +
					(KEY * CTA_THREADS) +
					((KEY * CTA_THREADS) >> LOG_MEM_BANKS)) :
				(threadIdx.x + (KEY * CTA_THREADS));

			items[KEY] = buffer[gather_offset];
		}
	}


	/**
	 * Decodes given keys to lookup digit offsets in shared memory
	 */
	__device__ __forceinline__ void DecodeDigitOffsets(
		UnsignedBits 	twiddled_keys[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD])
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			// Decode address of bin-offset in smem
			UnsignedBits digit = util::BFE(twiddled_keys[KEY], current_bit, RADIX_BITS);

			// Lookup base digit offset from shared memory
			digit_offsets[KEY] = smem_storage.digit_offsets[digit];
		}
	}

	/**
	 * Load tile of keys from global memory
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void LoadKeys(
		UnsignedBits 	keys[KEYS_PER_THREAD],
		SizeT 			tex_offset,
		const SizeT 	&guarded_elements)
	{

		if ((LOAD_MODIFIER == util::io::ld::tex) && FULL_TILE)
		{
			// Unguarded loads through tex
			#pragma unroll
			for (int PACK = 0; PACK < THREAD_TEX_LOADS; PACK++)
			{
                // Load tex vector
				UnsignedBits pack[ELEMENTS_PER_TEX];
				*reinterpret_cast<KeyTexType*>(&pack) = tex1Dfetch(
					TexKeys<KeyTexType>::ref,
					tex_offset + (threadIdx.x * THREAD_TEX_LOADS) + PACK);

				#pragma unroll
				for (int KEY = 0; KEY < ELEMENTS_PER_TEX; KEY++)
				{
					keys[(PACK * ELEMENTS_PER_TEX) + KEY] = pack[KEY];
				}
			}
		}
		else
		{
			// We have a partial-tile.  Need to perform guarded loads.
			#pragma unroll
			for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
			{
				int thread_offset = (threadIdx.x * KEYS_PER_THREAD) + KEY;

				keys[KEY] = (thread_offset < guarded_elements) ?
					*(d_in_keys + (tex_offset * ELEMENTS_PER_TEX) + thread_offset) :
					MAX_KEY;
			}
		}
	}


	/**
	 * Load tile of values from global memory
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void LoadValues(
		ValueType 		values[KEYS_PER_THREAD],
		SizeT 			tex_offset,
		const SizeT 	&guarded_elements)
	{
		if ((LOAD_MODIFIER == util::io::ld::tex) &&
			(util::NumericTraits<ValueType>::BUILT_IN) &&
			FULL_TILE)
		{
			// Unguarded loads through tex
			#pragma unroll
			for (int PACK = 0; PACK < THREAD_TEX_LOADS; PACK++)
			{
                // Load tex vector
				ValueType pack[ELEMENTS_PER_TEX];
				*reinterpret_cast<ValueTexType*>(&pack) = tex1Dfetch(
					TexValues<ValueTexType>::ref,
					tex_offset + (threadIdx.x * THREAD_TEX_LOADS) + PACK);

				#pragma unroll
				for (int KEY = 0; KEY < ELEMENTS_PER_TEX; KEY++)
				{
					values[(PACK * ELEMENTS_PER_TEX) + KEY] = pack[KEY];
				}
			}
		}
		else
		{
			// We have a partial-tile.  Need to perform guarded loads.
			#pragma unroll
			for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
			{
				int thread_offset = (threadIdx.x * KEYS_PER_THREAD) + KEY;

				if (thread_offset < guarded_elements)
				{
					values[KEY] = *(d_in_values + (tex_offset * ELEMENTS_PER_TEX) + thread_offset);
				}
			}
		}
	}


	/**
	 * Scatter ranked keys to global memory
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void ScatterKeys(
		UnsignedBits 	twiddled_keys[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD],		// (out parameter)
		unsigned int 	ranks[KEYS_PER_THREAD],
		SizeT 			guarded_elements)
	{
		if (SCATTER_STRATEGY == SCATTER_DIRECT)
		{
			// Scatter keys directly to global memory

			// Compute scatter offsets
			DecodeDigitOffsets(twiddled_keys, digit_offsets);

			// Untwiddle keys before outputting
			UnsignedBits keys[KEYS_PER_THREAD];
			TwiddleKeys<KeyTraits<KeyType>::TwiddleOut>(twiddled_keys, keys);

			// Scatter to global
			Iterate<0, KEYS_PER_THREAD>::template ScatterGlobal<FULL_TILE>(
				keys,
				ranks,
				digit_offsets,
				d_out_keys,
				guarded_elements);
		}
		else if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE)
		{
			// Use warp-aligned scattering of sorted keys from shared memory

			// Untwiddle keys before outputting
			UnsignedBits keys[KEYS_PER_THREAD];
			TwiddleKeys<KeyTraits<KeyType>::TwiddleOut>(twiddled_keys, keys);

			// Scatter to shared memory first
			ScatterRanked(ranks, keys, smem_storage.key_exchange);

			__syncthreads();

			// Gather sorted keys from smem and scatter to global using warp-aligned scattering
			Iterate<0, SCATTER_PASSES>::AlignedScatterPass(
				smem_storage,
				smem_storage.key_exchange,
				d_out_keys,
				guarded_elements);
		}
		else
		{
			// Normal two-phase scatter: exchange through shared memory, then
			// scatter sorted keys to global

			// Scatter to shared memory first (for better write-coalescing during global scatter)
			ScatterRanked(ranks, twiddled_keys, smem_storage.key_exchange);

			__syncthreads();

			// Gather sorted keys from shared memory
			GatherShared(twiddled_keys, smem_storage.key_exchange);

			// Compute scatter offsets
			DecodeDigitOffsets(twiddled_keys, digit_offsets);

			// Untwiddle keys before outputting
			UnsignedBits keys[KEYS_PER_THREAD];
			TwiddleKeys<KeyTraits<KeyType>::TwiddleOut>(twiddled_keys, keys);

			// Scatter keys to global memory
			Iterate<0, KEYS_PER_THREAD>::template ScatterGlobal<FULL_TILE>(
				keys,
				digit_offsets,
				d_out_keys,
				guarded_elements);
		}
	}


	/**
	 * Truck along associated values
	 */
	template <bool FULL_TILE, typename _ValueType>
	__device__ __forceinline__ void GatherScatterValues(
		_ValueType 		values[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD],
		unsigned int 	ranks[KEYS_PER_THREAD],
		SizeT 			tex_offset,
		SizeT 			guarded_elements)
	{
		// Load tile of values
		LoadValues<FULL_TILE>(values, tex_offset, guarded_elements);

		if (SCATTER_STRATEGY == SCATTER_DIRECT)
		{
			// Scatter values directly to global memory
			Iterate<0, KEYS_PER_THREAD>::template ScatterGlobal<FULL_TILE>(
				values,
				ranks,
				digit_offsets,
				d_out_values,
				guarded_elements);
		}
		else if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE)
		{
			__syncthreads();

			// Exchange values through shared memory for better write-coalescing
			ScatterRanked(ranks, values, smem_storage.value_exchange);

			__syncthreads();

			// Use explicitly warp-aligned scattering of values from shared memory
			Iterate<0, SCATTER_PASSES>::AlignedScatterPass(
				smem_storage,
				smem_storage.value_exchange,
				d_out_values,
				guarded_elements);
		}
		else
		{
			__syncthreads();

			// Exchange values through shared memory for better write-coalescing
			ScatterRanked(ranks, values, smem_storage.value_exchange);

			__syncthreads();

			// Gather values from shared
			GatherShared(values, smem_storage.value_exchange);

			// Scatter to global memory
			Iterate<0, KEYS_PER_THREAD>::template ScatterGlobal<FULL_TILE>(
				values,
				digit_offsets,
				d_out_values,
				guarded_elements);
		}
	}


	/**
	 * Truck along associated values (specialized for key-only sorting)
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void GatherScatterValues(
		util::NullType	values[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD],
		unsigned int 	ranks[KEYS_PER_THREAD],
		SizeT 			tex_offset,
		SizeT 			guarded_elements)
	{}


	/**
	 * Process tile
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void ProcessTile(
		SizeT tex_offset,
		const SizeT &guarded_elements = TILE_ELEMENTS)
	{
		// Per-thread tile data
		UnsignedBits 	keys[KEYS_PER_THREAD];					// Keys
		UnsignedBits 	twiddled_keys[KEYS_PER_THREAD];			// Twiddled (if necessary) keys
		ValueType 		values[KEYS_PER_THREAD];				// Values
		unsigned int	ranks[KEYS_PER_THREAD];					// For each key, the local rank within the CTA
		SizeT 			digit_offsets[KEYS_PER_THREAD];			// For each key, the global scatter base offset of the corresponding digit

		// Load tile of keys and twiddle bits if necessary
		LoadKeys<FULL_TILE>(keys, tex_offset, guarded_elements);

		__syncthreads();

		// Twiddle keys
		TwiddleKeys<KeyTraits<KeyType>::TwiddleIn>(keys, twiddled_keys);

		// Rank the twiddled keys
		CtaRadixRank::RankKeys(
			smem_storage.ranking_storage,
			twiddled_keys,
			ranks,
			smem_storage.digit_prefixes,
			current_bit);

		__syncthreads();

		// Update global scatter base offsets for each digit
		if ((CTA_THREADS == RADIX_DIGITS) || (threadIdx.x < RADIX_DIGITS))
		{
			my_digit_offset -= smem_storage.digit_prefixes[threadIdx.x];
			smem_storage.digit_offsets[threadIdx.x] = my_digit_offset;
			my_digit_offset += smem_storage.digit_prefixes[threadIdx.x + 1];
		}

		__syncthreads();

		// Scatter keys
		ScatterKeys<FULL_TILE>(twiddled_keys, digit_offsets, ranks, guarded_elements);

		// Gather/scatter values
		GatherScatterValues<FULL_TILE>(values, digit_offsets, ranks, tex_offset, guarded_elements);
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(const SizeT &guarded_elements)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT tex_offset = smem_storage.tex_offset;

		// Process full tiles of tile_elements
		while (tex_offset < smem_storage.tex_offset_limit)
		{
			ProcessTile<true>(tex_offset);
			tex_offset += TILE_TEX_LOADS;
		}

		// Clean up last partial tile with guarded-io
		if (guarded_elements)
		{
			ProcessTile<false>(tex_offset, guarded_elements);
		}
	}
};


} // namespace downsweep
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
