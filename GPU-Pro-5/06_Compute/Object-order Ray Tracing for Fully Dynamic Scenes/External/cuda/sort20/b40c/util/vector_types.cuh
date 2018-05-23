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
 * Utility code for working with vector types of arbitary typenames
 ******************************************************************************/

#pragma once

#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {


/**
 * Specializations of this vector-type template can be used to indicate the 
 * proper vector type for a given typename and vector size. We can use the ::Type
 * typedef to declare and work with the appropriate vectorized type for a given 
 * typename T.
 * 
 * For example, consider the following copy kernel that uses vec-2 loads 
 * and stores:
 * 
 * 		template <typename T>
 * 		__global__ void CopyKernel(T *d_in, T *d_out) 
 * 		{
 * 			typedef typename VecType<T, 2>::Type Vector;
 *
 * 			Vector datum;
 * 
 * 			Vector *d_in_v2 = (Vector *) d_in;
 * 			Vector *d_out_v2 = (Vector *) d_out;
 * 
 * 			datum = d_in_v2[threadIdx.x];
 * 			d_out_v2[threadIdx.x] = datum;
 * 		} 
 * 
 */
template <typename T, int vec_elements> struct VecType
{
	typedef util::NullType Type;
};

/**
 * Partially-specialized generic vec1 type 
 */
template <typename T> 
struct VecType<T, 1> {
	T x;
	typedef VecType<T, 1> Type;
};

/**
 * Partially-specialized generic vec2 type 
 */
template <typename T> 
struct VecType<T, 2> {
	T x;
	T y;
	typedef VecType<T, 2> Type;
};

/**
 * Partially-specialized generic vec4 type 
 */
template <typename T> 
struct VecType<T, 4> {
	T x;
	T y;
	T z;
	T w;
	typedef VecType<T, 4> Type;
};


/**
 * Macro for expanding partially-specialized built-in vector types
 */
#define CUB_DEFINE_VECTOR_TYPE(base_type, short_type)							\
  template<> struct VecType<base_type, 1> 										\
  {																				\
	typedef short_type##1 Type; 												\
  };      																		\
																				\
  template<> struct VecType<base_type, 2> 										\
  {																				\
	typedef short_type##2 Type; 												\
  };      																		\
																				\
  template<> struct VecType<base_type, 4> 										\
  {																				\
	typedef short_type##4 Type; 												\
  };


CUB_DEFINE_VECTOR_TYPE(char,               char)
CUB_DEFINE_VECTOR_TYPE(signed char,        char)
CUB_DEFINE_VECTOR_TYPE(short,              short)
CUB_DEFINE_VECTOR_TYPE(int,                int)
CUB_DEFINE_VECTOR_TYPE(long,               long)
CUB_DEFINE_VECTOR_TYPE(long long,          longlong)
CUB_DEFINE_VECTOR_TYPE(unsigned char,      uchar)
CUB_DEFINE_VECTOR_TYPE(unsigned short,     ushort)
CUB_DEFINE_VECTOR_TYPE(unsigned int,       uint)
CUB_DEFINE_VECTOR_TYPE(unsigned long,      ulong)
CUB_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
CUB_DEFINE_VECTOR_TYPE(float,              float)
CUB_DEFINE_VECTOR_TYPE(double,             double)


#undef CUB_DEFINE_VECTOR_TYPE


} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
