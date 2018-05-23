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
 *  Storage wrapper for multi-pass stream transformations that require a
 *  secondary problem storage array to stream results back and forth from.
 ******************************************************************************/

#pragma once

#include "../util/basic_utils.cuh"
#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {


/**
 * Storage wrapper for multi-pass stream transformations that require a
 * more than one problem storage array to stream results back and forth from.
 * 
 * This wrapper provides maximum flexibility for re-using device allocations
 * for subsequent transformations.  As such, it is the caller's responsibility
 * to free any non-NULL storage arrays when no longer needed.
 * 
 * Many multi-pass stream computations require at least two problem storage
 * arrays, e.g., one for reading in from, the other for writing out to.
 * (And their roles can be reversed for each subsequent pass.) This structure
 * tracks two sets of device vectors (a keys and a values sets), and a "selector"
 * member to index which vector in each set is "currently valid".  I.e., the
 * valid data within "MultiBuffer<2, int, int> b" is accessible by:
 * 
 * 		b.d_keys[b.selector];
 * 
 */
template <
	int BUFFER_COUNT,
	typename _KeyType,
	typename _ValueType = util::NullType>
struct MultiBuffer
{
	typedef _KeyType	KeyType;
	typedef _ValueType 	ValueType;

	// Set of device vector pointers for keys
	KeyType* d_keys[BUFFER_COUNT];
	
	// Set of device vector pointers for values
	ValueType* d_values[BUFFER_COUNT];

	// Selector into the set of device vector pointers (i.e., where the results are)
	int selector;

	// Constructor
	MultiBuffer()
	{
		selector = 0;
		for (int i = 0; i < BUFFER_COUNT; i++) {
			d_keys[i] = NULL;
			d_values[i] = NULL;
		}
	}
};



/**
 * Double buffer (a.k.a. page-flip, ping-pong, etc.) version of the
 * multi-buffer storage abstraction above.
 *
 * Many of the B40C primitives are templated upon the DoubleBuffer type: they
 * are compiled differently depending upon whether the declared type contains
 * keys-only versus key-value pairs (i.e., whether ValueType is util::NullType
 * or some real type).
 *
 * Declaring keys-only storage wrapper:
 *
 * 		DoubleBuffer<KeyType> key_storage;
 *
 * Declaring key-value storage wrapper:
 *
 * 		DoubleBuffer<KeyType, ValueType> key_value_storage;
 *
 */
template <
	typename KeyType,
	typename ValueType = util::NullType>
struct DoubleBuffer : MultiBuffer<2, KeyType, ValueType>
{
	typedef MultiBuffer<2, KeyType, ValueType> ParentType;

	// Constructor
	DoubleBuffer() : ParentType() {}

};


/**
 * Triple buffer version of the multi-buffer storage abstraction above.
 */
template <
	typename KeyType,
	typename ValueType = util::NullType>
struct TripleBuffer : MultiBuffer<3, KeyType, ValueType>
{
	typedef MultiBuffer<3, KeyType, ValueType> ParentType;

	// Constructor
	TripleBuffer() : ParentType() {}
};



} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
