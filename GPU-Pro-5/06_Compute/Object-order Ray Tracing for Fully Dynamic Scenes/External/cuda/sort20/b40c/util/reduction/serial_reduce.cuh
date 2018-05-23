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
 * Serial reduction over array types
 ******************************************************************************/

#pragma once

#include "../../util/operators.cuh"
#include "../../util/device_intrinsics.cuh"
#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {
namespace reduction {

//---------------------------------------------------------------------
// Helper functions for vectorizing reduction operations
//---------------------------------------------------------------------

template <typename T, typename ReductionOp>
__device__ __forceinline__ T VectorReduce(
	T a,
	T b,
	T c,
	ReductionOp reduction_op)
{
	return reduction_op(a, reduction_op(b, c));
}

template <>
__device__ __forceinline__ int VectorReduce<int, Sum<int> >(
	int a,
	int b,
	int c,
	Sum<int> reduction_op)
{
	return util::IADD3(a, b, c);
};

template <>
__device__ __forceinline__ unsigned int VectorReduce<unsigned int, Sum<unsigned int> >(
	unsigned int a,
	unsigned int b,
	unsigned int c,
	Sum<unsigned int> reduction_op)
{
	return util::IADD3(a, b, c);
};



/**
 * Have each thread perform a serial reduction over its specified segment
 */
template <int NUM_ELEMENTS>
struct SerialReduce
{
	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		template <typename T, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(T *partials, ReductionOp reduction_op)
		{
			T a = Iterate<COUNT - 2, TOTAL>::Invoke(partials, reduction_op);
			T b = partials[TOTAL - COUNT];
			T c = partials[TOTAL - (COUNT - 1)];

			return VectorReduce(a, b, c, reduction_op);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<2, TOTAL>
	{
		template <typename T, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(T *partials, ReductionOp reduction_op)
		{
			return reduction_op(partials[TOTAL - 2], partials[TOTAL - 1]);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<1, TOTAL>
	{
		template <typename T, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(T *partials, ReductionOp reduction_op)
		{
			return partials[TOTAL - 1];
		}
	};
	
	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Serial reduction with the specified operator
	 */
	template <typename T, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T *partials,
		ReductionOp reduction_op)
	{
		return Iterate<NUM_ELEMENTS, NUM_ELEMENTS>::Invoke(partials, reduction_op);
	}


	/**
	 * Serial reduction with the addition operator
	 */
	template <typename T>
	static __device__ __forceinline__ T Invoke(
		T *partials)
	{
		Sum<T> reduction_op;
		return Invoke(partials, reduction_op);
	}


	/**
	 * Serial reduction with the specified operator, seeded with the
	 * given exclusive partial
	 */
	template <typename T, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T *partials,
		T exclusive_partial,
		ReductionOp reduction_op)
	{
		return reduction_op(
			exclusive_partial,
			Invoke(partials, reduction_op));
	}

	/**
	 * Serial reduction with the addition operator, seeded with the
	 * given exclusive partial
	 */
	template <typename T, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T *partials,
		T exclusive_partial)
	{
		Sum<T> reduction_op;
		 return Invoke(partials, exclusive_partial, reduction_op);
	}
};


} // namespace reduction
} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
