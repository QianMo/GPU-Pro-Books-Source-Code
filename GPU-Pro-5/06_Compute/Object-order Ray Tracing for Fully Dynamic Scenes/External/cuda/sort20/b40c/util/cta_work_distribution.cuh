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
 * Work Management Datastructures
 ******************************************************************************/

#pragma once

#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {


/**
 * Structure for describing the limits of work-processing for a given CTA
 */
template <typename SizeT>
struct CtaWorkLimits
{
	SizeT 	offset;				// Offset at which this CTA begins processing
	SizeT 	elements;			// Total number of elements for this CTA to process
	SizeT 	guarded_offset; 	// Offset of final, partially-full tile (requires guarded loads)
	SizeT 	guarded_elements;	// Number of elements in partially-full tile
	SizeT 	out_of_bounds;		// Offset at which this CTA is out-of-bounds
	bool	last_block;			// If this block is the last block in the grid with any work

	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ CtaWorkLimits() {}

	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ CtaWorkLimits(
		SizeT 	offset,
		SizeT 	elements,
		SizeT 	guarded_offset,
		SizeT 	guarded_elements,
		SizeT 	out_of_bounds,
		bool	last_block) :
			offset(offset),
			elements(elements),
			guarded_offset(guarded_offset),
			guarded_elements(guarded_elements),
			out_of_bounds(out_of_bounds),
			last_block(last_block)
		{}
};


/**
 * Description of work distribution amongst CTAs
 *
 * A given threadblock may receive one of three different amounts of 
 * work: "big", "normal", and "last".  The big workloads are one
 * grain greater than the normal, and the last workload 
 * does the extra work.
 */
template <typename SizeT> 				// Integer type for indexing into problem arrays (e.g., int, long long, etc.)
struct CtaWorkDistribution
{
	SizeT 	num_elements;				// Number of elements in the problem
	SizeT 	total_grains;				// Number of "grain" blocks to break the problem into (round up)
	SizeT 	grains_per_cta;				// Number of "grain" blocks per CTA
	SizeT 	extra_grains;				// Number of CTAs having one extra "grain block"
	int 	grid_size;					// Number of CTAs
	int 	log_schedule_granularity;	// Scheduling grain

	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ CtaWorkDistribution(
		SizeT num_elements,
		int grid_size,
		int log_schedule_granularity)
	{
		this->log_schedule_granularity 	= log_schedule_granularity;
		int schedule_granularity 		= 1 << log_schedule_granularity;
		this->num_elements 				= num_elements;
		this->total_grains 				= ((num_elements + schedule_granularity - 1) >> log_schedule_granularity);	// round up
		this->grains_per_cta 			= total_grains / grid_size;													// round down for the ks
		this->extra_grains 				= total_grains - (grains_per_cta * grid_size);								// the CTAs with +1 grains
		this->grid_size 				= grid_size;
	}


	/**
	 * Computes work limits for the current CTA
	 */	
	__host__ __device__ __forceinline__ void GetCtaWorkLimits(
		CtaWorkLimits<SizeT> &work_limits,		// Out param
		int log_tile_elements)
	{
		const int TILE_ELEMENTS = 1 << log_tile_elements;
		
		// Compute number of elements and offset at which to start tile processing
		if (blockIdx.x < extra_grains) {

			// This CTA gets grains_per_cta+1 grains
			work_limits.elements = (grains_per_cta + 1) << log_schedule_granularity;
			work_limits.offset = work_limits.elements * blockIdx.x;

		} else if (blockIdx.x < total_grains) {

			// This CTA gets grains_per_cta grains
			work_limits.elements = grains_per_cta << log_schedule_granularity;
			work_limits.offset = (work_limits.elements * blockIdx.x) + (extra_grains << log_schedule_granularity);

		} else {

			// This CTA gets no work (problem small enough that some CTAs don't even a single grain)
			work_limits.elements = 0;
			work_limits.offset = 0;
		}

		// The offset at which this CTA is out-of-bounds
		work_limits.out_of_bounds = work_limits.offset + work_limits.elements;

		// Correct for the case where the last CTA having work has rounded its last grain up past the end
		if (work_limits.last_block = work_limits.out_of_bounds >= num_elements) {
			work_limits.out_of_bounds = num_elements;
			work_limits.elements = num_elements - work_limits.offset;
		}

		// The number of extra guarded-load elements to process afterward (always
		// less than a full tile)
		work_limits.guarded_elements = work_limits.elements & (TILE_ELEMENTS - 1);

		// The tile-aligned limit for full-tile processing
		work_limits.guarded_offset = work_limits.out_of_bounds - work_limits.guarded_elements;
	}


	/**
	 * Print to stdout
	 */
	void Print()
	{
		unsigned long schedule_granularity = 1 << log_schedule_granularity;
		unsigned long last_grain_elements = num_elements & (schedule_granularity - 1);

		if (last_grain_elements == 0) last_grain_elements = schedule_granularity;

		printf("num_elements: %lu, "
			"total_grains: %lu, "
			"schedule_granularity: %lu, "
			"grains_per_cta: %lu, "
			"extra_grains: %lu, "
			"grid_size: %lu, "
			"last_grain_elements: %lu\n",
				(unsigned long) num_elements,
				(unsigned long) total_grains,
				(unsigned long) schedule_granularity,
				(unsigned long) grains_per_cta,
				(unsigned long) extra_grains,
				(unsigned long) grid_size,
				(unsigned long) last_grain_elements);
	}
};


} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
