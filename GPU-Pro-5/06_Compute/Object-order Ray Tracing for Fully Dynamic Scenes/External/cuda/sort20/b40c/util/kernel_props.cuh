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
 * Kernel function properties
 ******************************************************************************/

#pragma once

#include <stdlib.h>

#include <map>

#include "../util/cuda_properties.cuh"
#include "../util/error_utils.cuh"
#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {

/**
 * Kernel function properties
 */
struct KernelProps
{
	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	int 									threads;
	int										sm_arch;
	int										sm_count;
	cudaFuncAttributes 						kernel_attrs;
	int 									max_cta_occupancy;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Initializer
	 */
	template <typename KernelFunc>
	cudaError_t Init(
		KernelFunc kernel_func,
		int threads,
		int sm_arch,
		int sm_count)
	{
		cudaError_t error = cudaSuccess;

		do {
			// Initialize fields
			this->threads = threads;
			this->sm_arch = sm_arch;
			this->sm_count = sm_count;

			// Get kernel attributes
			error = util::B40CPerror(
				cudaFuncGetAttributes(&kernel_attrs, kernel_func),
				"cudaFuncGetAttributes failed",
				__FILE__,
				__LINE__);
			if (error) break;

			// Compute SM CTA occupancy by resource
			int max_block_occupancy = CUB_SM_CTAS(sm_arch);
			int max_thread_occupancy = CUB_SM_THREADS(sm_arch) / threads;
			int max_smem_occupancy = (kernel_attrs.sharedSizeBytes > 0) ?
					(CUB_SMEM_BYTES(sm_arch) / kernel_attrs.sharedSizeBytes) :
					max_block_occupancy;
			int max_reg_occupancy = (kernel_attrs.sharedSizeBytes > 0) ?
					CUB_SM_REGISTERS(sm_arch) / (kernel_attrs.numRegs * threads) :
					max_block_occupancy;

			// Determine overall SM CTA occupancy
			max_cta_occupancy = CUB_MIN(
				CUB_MIN(max_block_occupancy, max_thread_occupancy),
				CUB_MIN(max_smem_occupancy, max_reg_occupancy));

		} while (0);

		return error;
	}


	/**
	 * Returns the number of threadblocks to launch for the given problem size.
	 * May over/under subscribe the current device based upon heuristics.  Does not
	 * the optional max_grid_size limit.
	 *
	 * Useful for kernels that evenly divide up the work amongst threadblocks.
	 */
	int OversubscribedGridSize(
		int schedule_granularity,
		int num_elements,
		int max_grid_size = 0) const
	{
		int grid_size;
		int grains = (num_elements + schedule_granularity - 1) / schedule_granularity;

		if (sm_arch < 120)
		{
			// G80/G90: double CTA occupancy times SM count
			grid_size = 2 * max_cta_occupancy * sm_count;

		}
		else if (sm_arch < 200)
		{
			// GT200: Special sauce.  Start with with full occupancy of all SMs
			grid_size = max_cta_occupancy * sm_count;

			int bumps = 0;
			double cutoff = 0.005;

			while (true) {

				double quotient = double(num_elements) /
					grid_size /
					schedule_granularity;

				int log = log2(quotient) + 0.5;

				int primary = (1 << log) *
					grid_size *
					schedule_granularity;

				double ratio = double(num_elements) / primary;

				if (((ratio < 1.00 + cutoff) && (ratio > 1.00 - cutoff)) ||
					((ratio < 0.75 + cutoff) && (ratio > 0.75 - cutoff)) ||
					((ratio < 0.50 + cutoff) && (ratio > 0.50 - cutoff)) ||
					((ratio < 0.25 + cutoff) && (ratio > 0.25 - cutoff)))
				{
					if (bumps == 3) {
						// Bump it up by 33
						grid_size += 33;
						bumps = 0;
					} else {
						// Bump it down by 1
						grid_size--;
						bumps++;
					}
					continue;
				}

				break;
			}
		}
		else if (sm_arch < 300)
		{
			// Fermi: quadruple CTA occupancy times SM count
			grid_size = 4 * max_cta_occupancy * sm_count;
		}
		else
		{
			// Kepler: quadruple CTA occupancy times SM count
			grid_size = 4 * max_cta_occupancy * sm_count;
		}

		grid_size = (max_grid_size > 0) ? max_grid_size : grid_size;	// Apply override, if specified
		grid_size = CUB_MIN(grains, grid_size);							// Floor to the number of schedulable grains

		return grid_size;
	}


	/**
	 * Return dynamic padding to reduce occupancy to a multiple of the specified base_occupancy
	 */
	int SmemPadding(int base_occupancy) const
	{
		div_t div_result = div(max_cta_occupancy, base_occupancy);
		if ((!div_result.quot) || (!div_result.rem)) {
			return 0;													// Perfect division (or cannot be padded)
		}

		int target_occupancy = div_result.quot * base_occupancy;
		int required_shared = CUB_SMEM_BYTES(sm_arch) / target_occupancy;
		int padding = (required_shared - kernel_attrs.sharedSizeBytes) / 128 * 128;					// Round down to nearest 128B

		return padding;
	}
};





} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
