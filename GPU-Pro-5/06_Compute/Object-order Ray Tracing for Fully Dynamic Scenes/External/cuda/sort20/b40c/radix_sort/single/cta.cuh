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
 * CTA-processing functionality for single-CTA radix sort kernel
 ******************************************************************************/

#pragma once

#include "../../util/basic_utils.cuh"
#include "../../util/io/modified_load.cuh"
#include "../../util/io/modified_store.cuh"
#include "../../util/ns_umbrella.cuh"

#include "../../radix_sort/sort_utils.cuh"
#include "../../radix_sort/cta_radix_sort.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace single {


/**
 * Partitioning downsweep scan CTA
 */
template <
	typename KernelPolicy,
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

	enum
	{
		RADIX_BITS					= KernelPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,

		KEYS_ONLY 					= util::Equals<ValueType, util::NullType>::VALUE,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		CTA_THREADS 				= KernelPolicy::CTA_THREADS,
		WARPS						= CTA_THREADS / WARP_THREADS,

		KEYS_PER_THREAD				= KernelPolicy::THREAD_ELEMENTS,
		TILE_ELEMENTS				= KernelPolicy::TILE_ELEMENTS,

		LOG_MEM_BANKS				= CUB_LOG_MEM_BANKS(__CUB_CUDA_ARCH__),
		MEM_BANKS					= 1 << LOG_MEM_BANKS,

		// Insert padding if the number of keys per thread is a power of two
		PADDING  					= ((KEYS_PER_THREAD & (KEYS_PER_THREAD - 1)) == 0),
	};


	// CtaRadixSort utility type
	typedef CtaRadixSort<
		UnsignedBits,
		CTA_THREADS,
		KEYS_PER_THREAD,
		RADIX_BITS,
		ValueType,
		KernelPolicy::SMEM_CONFIG> CtaRadixSort;

	// Texture types
	typedef Textures<KeyType, ValueType, 1> 			Textures;
	typedef typename Textures::KeyTexType 				KeyTexType;
	typedef typename Textures::ValueTexType 			ValueTexType;

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		typename CtaRadixSort::SmemStorage sorting_storage;
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 *
	 */
	template <typename TexT, typename TexRef, typename SizeT>
	static __device__ __forceinline__ void LoadTile(
		TexT 			*items,
		TexT 			*d_in,
		TexRef			tex_ref,
		const SizeT		&cta_offset,
		const int 		&guarded_elements)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			// Load item from texture if in bounds
			int global_offset = (threadIdx.x * KEYS_PER_THREAD) + KEY;
			if (global_offset < guarded_elements)
			{
				if (LOAD_MODIFIER == util::io::ld::tex)
				{
					items[KEY] = tex1Dfetch(
						tex_ref,
						cta_offset + global_offset);
				}
				else
				{
					items[KEY] = d_in[cta_offset + global_offset];
				}
			}
		}
	}

	/**
	 *
	 */
	template <typename T, typename SizeT>
	static __device__ __forceinline__ void StoreTile(
		T 				*items,
		T 				*d_out,
		const SizeT		&cta_offset,
		const int 		&guarded_elements)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int global_offset = (KEY * CTA_THREADS) + threadIdx.x;
			if (global_offset < guarded_elements)
			{
				d_out[cta_offset + global_offset] = items[KEY];
			}
		}
	}


	/**
	 * ProcessTile.  (Specialized for keys-only sorting.)
	 */
	template <typename SizeT>
	static __device__ __forceinline__ void ProcessTile(
		SmemStorage 	&smem_storage,
		KeyType 		*d_keys,
		util::NullType 	*d_values,
		unsigned int 	current_bit,
		unsigned int 	bits_remaining,
		const SizeT		&cta_offset,
		const int 		&guarded_elements)
	{
		UnsignedBits keys[KEYS_PER_THREAD];

		// Initialize keys to default key value
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = MAX_KEY;
		}

		// Load keys
		LoadTile(
			reinterpret_cast<KeyTexType*>(keys),
			reinterpret_cast<KeyTexType*>(d_keys),
			TexKeys<KeyTexType>::ref,
			cta_offset,
			guarded_elements);

		// Twiddle key bits if necessary
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = KeyTraits<KeyType>::TwiddleIn(keys[KEY]);
		}

		// Sort
		CtaRadixSort::SortThreadToCtaStride(
			smem_storage.sorting_storage,
			keys,
			current_bit,
			bits_remaining);

		// Twiddle key bits if necessary
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = KeyTraits<KeyType>::TwiddleOut(keys[KEY]);
		}

		// Store keys
		StoreTile(
			keys,
			reinterpret_cast<UnsignedBits*>(d_keys),
			cta_offset,
			guarded_elements);
	}


	/**
	 * ProcessTile.  (Specialized for keys-value sorting.)
	 */
	template <typename SizeT, typename _ValueType>
	static __device__ __forceinline__ void ProcessTile(
		SmemStorage 	&smem_storage,
		KeyType 		*d_keys,
		_ValueType		*d_values,
		unsigned int 	current_bit,
		unsigned int 	bits_remaining,
		const SizeT		&cta_offset,
		const int 		&guarded_elements)
	{
		UnsignedBits 	keys[KEYS_PER_THREAD];
		ValueType 		values[KEYS_PER_THREAD];

		// Initialize keys to default key value
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = MAX_KEY;
		}

		// Load keys
		LoadTile(
			reinterpret_cast<KeyTexType*>(keys),
			reinterpret_cast<KeyTexType*>(d_keys),
			TexKeys<KeyTexType>::ref,
			cta_offset,
			guarded_elements);

		// Twiddle key bits if necessary
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = KeyTraits<KeyType>::TwiddleIn(keys[KEY]);
		}

		// Load values
		LoadTile(
			reinterpret_cast<ValueTexType*>(values),
			reinterpret_cast<ValueTexType*>(d_values),
			TexValues<ValueTexType>::ref,
			cta_offset,
			guarded_elements);

		// Sort
		CtaRadixSort::SortThreadToCtaStride(
			smem_storage.sorting_storage,
			keys,
			values,
			current_bit,
			bits_remaining);

		// Twiddle key bits if necessary
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = KeyTraits<KeyType>::TwiddleOut(keys[KEY]);
		}

		// Store keys
		StoreTile(
			keys,
			reinterpret_cast<UnsignedBits*>(d_keys),
			cta_offset,
			guarded_elements);

		// Store values
		StoreTile(
			values,
			d_values,
			cta_offset,
			guarded_elements);
	}



};


} // namespace single
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
