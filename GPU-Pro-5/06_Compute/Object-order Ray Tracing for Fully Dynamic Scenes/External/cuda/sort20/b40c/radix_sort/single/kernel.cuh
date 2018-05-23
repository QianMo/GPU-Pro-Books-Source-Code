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
 * Radix sort single-CTA sort kernel
 ******************************************************************************/

#pragma once

#include "../../radix_sort/single/cta.cuh"
#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace single {


/**
 * Radix sort single-CTA sort kernel entry point
 */
template <
	typename KernelPolicy,
	typename KeyType,
	typename ValueType>
__launch_bounds__ (KernelPolicy::CTA_THREADS, 1)
__global__ 
void Kernel(
	KeyType 							*d_keys,
	ValueType 							*d_values,
	unsigned int 						current_bit,
	unsigned int						bits_remaining,
	unsigned int 						num_elements)
{
	// CTA abstraction type
	typedef Cta<KernelPolicy, KeyType, ValueType> Cta;

	// Shared memory pool
	__shared__ typename Cta::SmemStorage smem_storage;

	Cta::ProcessTile(
		smem_storage,
		d_keys,
		d_values,
		current_bit,
		bits_remaining,
		0,
		num_elements);
}



} // namespace single
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
