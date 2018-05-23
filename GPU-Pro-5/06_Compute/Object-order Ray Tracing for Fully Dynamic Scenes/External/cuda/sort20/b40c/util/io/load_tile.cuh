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
 * Kernel utilities for loading tiles of data through global memory
 * with cache modifiers
 ******************************************************************************/

#pragma once

#include "../../util/operators.cuh"
#include "../../util/vector_types.cuh"
#include "../../util/io/modified_load.cuh"
#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {
namespace io {


/**
 * Static operator wrapping structure.
 */
template <typename T, typename R = T>
struct Operators
{
	/**
	 * Empty default transform function
	 */
	static __device__ __forceinline__ void NopTransform(T &val) {}

};


/**
 * Load a tile of items
 */
template <
	int LOG_LOADS_PER_TILE, 									// Number of vector loads (log)
	int LOG_LOAD_VEC_SIZE,										// Number of items per vector load (log)
	int ACTIVE_THREADS,											// Active threads that will be loading
	ld::CacheModifier CACHE_MODIFIER,							// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
	bool CHECK_ALIGNMENT>										// Whether or not to check alignment to see if vector loads can be used
struct LoadTile
{
	enum {
		LOADS_PER_TILE 				= 1 << LOG_LOADS_PER_TILE,
		LOAD_VEC_SIZE 				= 1 << LOG_LOAD_VEC_SIZE,
		LOG_ELEMENTS_PER_THREAD		= LOG_LOADS_PER_TILE + LOG_LOAD_VEC_SIZE,
		ELEMENTS_PER_THREAD			= 1 << LOG_ELEMENTS_PER_THREAD,
		TILE_SIZE 					= ACTIVE_THREADS * ELEMENTS_PER_THREAD,
	};
	
	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	template <int CURRENT, int STRIDE>
	struct IterateVecLoad
	{
		// Vector (unguarded)
		template <typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			ModifiedLoad<CACHE_MODIFIER>::Ld(
				vectors[CURRENT],
				d_in_vectors + CURRENT);

			IterateVecLoad<CURRENT + 1, STRIDE>::LoadVector(
				vectors,
				d_in_vectors);

		}
	};

	template <int STRIDE>
	struct IterateVecLoad<STRIDE, STRIDE>
	{
		// Vector (unguarded)
		template <typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			VectorType vectors[],
			VectorType *d_in_vectors) {}
	};

	/**
	 * Next vec element of a vector-load
	 */
	template <int LOAD, int VEC, int dummy = 0>
	struct Iterate
	{
		// Vector (unguarded)
		template <
			typename T,
			void Transform(T&),
			int STRIDE,
			typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			if (VEC == 0) {
				// Load all vectors for this load
				IterateVecLoad<0, STRIDE>::LoadVector(vectors, d_in_vectors);
			}

			Transform(data[LOAD][VEC]);

			Iterate<LOAD, VEC + 1>::template LoadVector<T, Transform, STRIDE>(
				data, vectors, d_in_vectors);
		}

		// Regular (unguarded)
		template <typename T, void Transform(T&)>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in)
		{
			int thread_offset = (threadIdx.x * LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);
			Transform(data[LOAD][VEC]);

			Iterate<LOAD, VEC + 1>::template LoadValid<T, Transform>(
				data, d_in);
		}

		// Regular (guarded)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x * LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);
				Transform(data[LOAD][VEC]);
			}

			Iterate<LOAD, VEC + 1>::template LoadValid<T, Transform>(
				data, d_in, guarded_elements);
		}

		// Regular (guarded with out-of-bounds default)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T oob_default,
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x * LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);
				Transform(data[LOAD][VEC]);
			} else {
				data[LOAD][VEC] = oob_default;
			}

			Iterate<LOAD, VEC + 1>::template LoadValid<T, Transform>(
				data, oob_default, d_in, guarded_elements);
		}
	};


	/**
	 * Next load
	 */
	template <int LOAD, int dummy>
	struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// Vector (unguarded)
		template <
			typename T,
			void Transform(T&),
			int STRIDE,
			typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			Iterate<LOAD + 1, 0>::template LoadVector<T, Transform, STRIDE>(
				data,
				vectors + STRIDE,
				d_in_vectors + (ACTIVE_THREADS * STRIDE));
		}

		// Regular (unguarded)
		template <typename T, void Transform(T&)>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in)
		{
			Iterate<LOAD + 1, 0>::template LoadValid<T, Transform>(
				data, d_in);
		}

		// Regular (guarded)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			Iterate<LOAD + 1, 0>::template LoadValid<T, Transform>(
				data, d_in, guarded_elements);
		}

		// Regular (guarded with out-of-bounds default)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T oob_default,
			T *d_in,
			const SizeT &guarded_elements)
		{
			Iterate<LOAD + 1, 0>::template LoadValid<T, Transform>(
				data, oob_default, d_in, guarded_elements);
		}
	};
	
	/**
	 * Terminate
	 */
	template <int dummy>
	struct Iterate<LOADS_PER_TILE, 0, dummy>
	{
		// Vector (unguarded)
		template <typename T, void Transform(T&), int STRIDE, typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors) {}

		// Regular (unguarded)
		template <typename T, void Transform(T&)>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in) {}

		// Regular (guarded)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements) {}

		// Regular (guarded with out-of-bounds default)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T oob_default,
			T *d_in,
			const SizeT &guarded_elements) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Load a full tile with transform
	 */
	template <
		typename T,
		void Transform(T&),
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset)
	{
		const size_t MASK = ((sizeof(T) * 8 * LOAD_VEC_SIZE) - 1);

		if ((CHECK_ALIGNMENT) && (((size_t) d_in) & MASK)) {

			Iterate<0, 0>::template LoadValid<T, Transform>(
				data, d_in + cta_offset);

		} else if (LOAD_VEC_SIZE > 4) {

			// Use an aliased pointer to keys array to perform built-in vector loads
			typedef typename VecType<T, 4>::Type VectorType;

			VectorType *vectors = (VectorType *) data;
			VectorType *d_in_vectors = (VectorType *) (d_in + cta_offset + (threadIdx.x * LOAD_VEC_SIZE));

			Iterate<0, 0>::template LoadVector<T, Transform, (LOAD_VEC_SIZE / 4)>(
				data, vectors, d_in_vectors);

		} else {

			// Use an aliased pointer to keys array to perform built-in vector loads
			typedef typename VecType<T, LOAD_VEC_SIZE>::Type VectorType;

			VectorType *vectors = (VectorType *) data;
			VectorType *d_in_vectors = (VectorType *) (d_in + cta_offset + (threadIdx.x * LOAD_VEC_SIZE));

			Iterate<0, 0>::template LoadVector<T, Transform, 1>(
				data, vectors, d_in_vectors);
		}
	}


	/**
	 * Load a full tile
	 */
	template <
		typename T,
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset)
	{
		LoadValid<T, Operators<T>::NopTransform>(data, d_in, cta_offset);
	}


	/**
	 * Load guarded_elements of a tile with transform and out-of-bounds default
	 */
	template <
		typename T,
		void Transform(T&),
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		T oob_default)
	{
		if (guarded_elements >= TILE_SIZE) {
			LoadValid<T, Transform>(data, d_in, cta_offset);
		} else {
			Iterate<0, 0>::template LoadValid<T, Transform>(
				data, oob_default, d_in + cta_offset, guarded_elements);
		}
	}


	/**
	 * Load guarded_elements of a tile with transform
	 */
	template <
		typename T,
		void Transform(T&),
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		if (guarded_elements >= TILE_SIZE) {
			LoadValid<T, Transform>(data, d_in, cta_offset);
		} else {
			Iterate<0, 0>::template LoadValid<T, Transform>(
				data, d_in + cta_offset, guarded_elements);
		}
	}


	/**
	 * Load guarded_elements of a tile and out_of_bounds default
	 */
	template <
		typename T,
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		T oob_default)
	{
		LoadValid<T, Operators<T>::NopTransform>(
			data, d_in, cta_offset, guarded_elements, oob_default);
	}


	/**
	 * Load guarded_elements of a tile
	 */
	template <
		typename T,
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		LoadValid<T, Operators<T>::NopTransform, int>(
			data, d_in, cta_offset, guarded_elements);
	}
};


} // namespace io
} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
