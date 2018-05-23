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
 * Radix sort spine scan kernel
 ******************************************************************************/

#pragma once

#include "../../radix_sort/spine/cta.cuh"
#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace spine {


/**
 * Consecutive removal spine scan kernel entry point
 */
template <
	typename KernelPolicy,
	typename T,
	typename SizeT>
__launch_bounds__ (KernelPolicy::CTA_THREADS, 1)
__global__ 
void Kernel(
	T			*d_in,
	T			*d_out,
	SizeT 		spine_elements)
{
	// CTA abstraction type
	typedef Cta<KernelPolicy, T, SizeT> Cta;

	// Shared memory pool
	__shared__ typename Cta::SmemStorage smem_storage;

	// Only CTA-0 needs to run
	if (blockIdx.x > 0) return;

	Cta cta(smem_storage, d_in, d_out);
	cta.ProcessWorkRange(spine_elements);
}

} // namespace spine
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
