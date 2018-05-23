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
 * Radix sort upsweep reduction kernel
 ******************************************************************************/

#pragma once

#include "../../radix_sort/upsweep/cta.cuh"
#include "../../util/cuda_properties.cuh"
#include "../../util/cta_work_distribution.cuh"
#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace upsweep {


/**
 * Radix sort upsweep reduction kernel entry point
 */
template <
	typename KernelPolicy,
	typename SizeT,
	typename KeyType>
__launch_bounds__ (KernelPolicy::CTA_THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	SizeT 		*d_spine,
	KeyType 	*d_in_keys,
	util::CtaWorkDistribution<SizeT> work_decomposition,
	unsigned int current_bit)
{
	// CTA abstraction type
	typedef Cta<KernelPolicy, SizeT, KeyType> Cta;

	// Shared memory pool
	__shared__ typename Cta::SmemStorage smem_storage;
	
	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.GetCtaWorkLimits(
		work_limits,
		KernelPolicy::LOG_TILE_ELEMENTS);

	Cta cta(smem_storage, d_spine, d_in_keys, current_bit);

	cta.ProcessWorkRange(work_limits);
}


} // namespace upsweep
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
