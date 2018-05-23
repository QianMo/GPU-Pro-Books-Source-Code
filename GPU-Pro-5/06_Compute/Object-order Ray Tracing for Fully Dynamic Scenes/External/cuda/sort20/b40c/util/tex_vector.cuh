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
 * Kernel utilities loading for CTA-sized tiles of data from memory
 ******************************************************************************/

#pragma once

#include "../util/numeric_traits.cuh"
#include "../util/vector_types.cuh"
#include "../util/io/modified_load.cuh"
#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {

/**
 * Texture vector types for reading ELEMENTS consecutive elements of T per thread
 */
template <typename T, int ELEMENTS, bool BUILT_IN = NumericTraits<T>::BUILT_IN>
struct TexVector
{
	enum {
		TEX_VEC_SIZE = (sizeof(T) > 4) ?
							(ELEMENTS % 2 == 1) ?			// 64-bit built-in types
								2 : 								// cast as vec-2 ints (odd)
								4 :									// cast as vec-4 ints (multiple of two)
							(ELEMENTS % 2 == 1) ?			// 32-bit built-in types
								1 : 								// vec-1 (odd)
								(ELEMENTS % 4 == 0) ?
									4 :								// vec-4 (multiple of 4)
									2,								// vec-2 (multiple of 2)
	};

	// Texture base type
	typedef typename If<(sizeof(T) > 4),
		unsigned int,							// use int for 64-bit built-in types
		T>::Type								// use T for other built-in types
			TexBase;

	// Texture vector type
	typedef typename util::VecType<TexBase, TEX_VEC_SIZE>::Type VecType;

	// Number of T loaded per texture load
	enum {
		ELEMENTS_PER_TEX = sizeof(VecType) / sizeof(T),
	};

	// Texture reference type
	typedef texture<TexVector, cudaTextureType1D, cudaReadModeElementType> TexRef;
};


/**
 * Dummy values specialized for non-built-in types
 */
template <typename T, int ELEMENTS>
struct TexVector<T, ELEMENTS, false>
{
	enum {
		TEX_VEC_SIZE 		= 1,
		ELEMENTS_PER_TEX 	= 1,
	};

	// Texture base type
	typedef char TexBase;

	// Texture vector type
	typedef typename util::VecType<TexBase, TEX_VEC_SIZE>::Type VecType;

	// Texture reference type
	typedef texture<TexVector, cudaTextureType1D, cudaReadModeElementType> TexRef;
};


} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
