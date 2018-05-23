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
 * raking Grid Description
 ******************************************************************************/

#pragma once

#include "../util/cuda_properties.cuh"
#include "../util/basic_utils.cuh"
#include "../util/numeric_traits.cuh"
#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {


/**
 * Description of a (typically) conflict-free serial-reduce-then-scan (raking) 
 * shared-memory grid.
 *
 * A "lane" for reduction/scan consists of one value (i.e., "partial") per
 * active thread.  A grid consists of one or more scan lanes. The lane(s)
 * can be sequentially "raked" by the specified number of raking threads
 * (e.g., for upsweep reduction or downsweep scanning), where each raking
 * thread progresses serially through a segment that is its share of the
 * total grid.
 *
 * Depending on how the raking threads are further reduced/scanned, the lanes
 * can be independent (i.e., only reducing the results from every
 * SEGS_PER_LANE raking threads), or fully dependent (i.e., reducing the
 * results from every raking thread)
 *
 * Must have at least as many raking threads as lanes (i.e., at least one
 * raking thread for each lane).
 *
 * If (there are prefix dependences between lanes) AND (more than one warp
 * of raking threads is specified), a secondary raking grid will
 * be typed-out in order to facilitate communication between warps of raking
 * threads.
 *
 * (N.B.: Typically two-level grids are a losing performance proposition)
 */
template <
	typename _T,									// Type of items we will be reducing/scanning
	int _LOG_ACTIVE_THREADS, 						// Number of threads placing a lane partial (i.e., the number of partials per lane)
	int _LOG_SCAN_LANES,							// Number of scan lanes
	int _LOG_RAKING_THREADS, 						// Number of threads used for raking (typically 1 warp)
	bool _DEPENDENT_LANES>							// If there are prefix dependences between lanes (i.e., downsweeping will incorporate aggregates from previous lanes)
struct RakingGrid
{
	// Type of items we will be reducing/scanning
	typedef _T T;
	
	// Warpscan type (using volatile storage for built-in types allows us to omit thread-fence
	// operations during warp-synchronous code)
	typedef typename util::If<
		(util::NumericTraits<T>::REPRESENTATION == int(util::NOT_A_NUMBER)),
		T,
		volatile T>::Type WarpscanT;

	// N.B.: We use an enum type here b/c of a NVCC-win compiler bug where the
	// compiler can't handle ternary expressions in static-const fields having
	// both evaluation targets as local const expressions.
	enum {
		LOG_WARP_THREADS 				= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS					= 1 << LOG_WARP_THREADS,

		// Number of scan lanes
		LOG_SCAN_LANES					= _LOG_SCAN_LANES,
		SCAN_LANES						= 1 <<LOG_SCAN_LANES,

		// Number number of partials per lane
		LOG_PARTIALS_PER_LANE 			= _LOG_ACTIVE_THREADS,
		PARTIALS_PER_LANE				= 1 << LOG_PARTIALS_PER_LANE,

		// Number of raking threads
		LOG_RAKING_THREADS				= _LOG_RAKING_THREADS,
		RAKING_THREADS					= 1 << LOG_RAKING_THREADS,

		// Number of raking threads per lane
		LOG_RAKING_THREADS_PER_LANE		= LOG_RAKING_THREADS - LOG_SCAN_LANES,			// must be positive!
		RAKING_THREADS_PER_LANE			= 1 << LOG_RAKING_THREADS_PER_LANE,

		// Partials to be raked per raking thread
		LOG_PARTIALS_PER_SEG 			= LOG_PARTIALS_PER_LANE - LOG_RAKING_THREADS_PER_LANE,
		PARTIALS_PER_SEG 				= 1 << LOG_PARTIALS_PER_SEG,

		// Number of partials that we can put in one stripe across the shared memory banks
		LOG_PARTIALS_PER_BANK_ARRAY		= CUB_LOG_MEM_BANKS(__CUB_CUDA_ARCH__) +
											CUB_LOG_BANK_STRIDE_BYTES(__CUB_CUDA_ARCH__) -
											Log2<sizeof(T)>::VALUE,
		PARTIALS_PER_BANK_ARRAY			= 1 << LOG_PARTIALS_PER_BANK_ARRAY,

		LOG_SEGS_PER_BANK_ARRAY 		= CUB_MAX(0, LOG_PARTIALS_PER_BANK_ARRAY - LOG_PARTIALS_PER_SEG),
		SEGS_PER_BANK_ARRAY				= 1 << LOG_SEGS_PER_BANK_ARRAY,

		// Whether or not one warp of raking threads can rake entirely in one stripe across the shared memory banks
		NO_PADDING = (LOG_SEGS_PER_BANK_ARRAY >= LOG_WARP_THREADS),

		// Number of raking segments we can have without padding (i.e., a "row")
		LOG_SEGS_PER_ROW 				= (NO_PADDING) ?
											LOG_RAKING_THREADS :												// All raking threads (segments)
											CUB_MIN(LOG_RAKING_THREADS_PER_LANE, LOG_SEGS_PER_BANK_ARRAY),		// Up to as many segments per lane (all lanes must have same amount of padding to have constant lane stride)
		SEGS_PER_ROW					= 1 << LOG_SEGS_PER_ROW,

		// Number of partials per row
		LOG_PARTIALS_PER_ROW			= LOG_SEGS_PER_ROW + LOG_PARTIALS_PER_SEG,
		PARTIALS_PER_ROW				= 1 << LOG_PARTIALS_PER_ROW,

		// Number of partials that we must use to "pad out" one memory bank
		LOG_BANK_PADDING_PARTIALS		= CUB_MAX(0, CUB_LOG_BANK_STRIDE_BYTES(__CUB_CUDA_ARCH__) - Log2<sizeof(T)>::VALUE),
		BANK_PADDING_PARTIALS			= 1 << LOG_BANK_PADDING_PARTIALS,

		// Number of partials that we must use to "pad out" a lane to one memory bank
		LANE_PADDING_PARTIALS			= CUB_MAX(0, PARTIALS_PER_BANK_ARRAY - PARTIALS_PER_LANE),

		// Number of partials (including padding) per "row"
		PADDED_PARTIALS_PER_ROW			= (NO_PADDING) ?
											PARTIALS_PER_ROW :
											PARTIALS_PER_ROW + LANE_PADDING_PARTIALS + BANK_PADDING_PARTIALS,

		// Number of rows in the grid
		LOG_ROWS						= LOG_RAKING_THREADS - LOG_SEGS_PER_ROW,
		ROWS 							= 1 << LOG_ROWS,

		// Number of rows per lane (always at least one)
		LOG_ROWS_PER_LANE				= CUB_MAX(0, LOG_RAKING_THREADS_PER_LANE - LOG_SEGS_PER_ROW),
		ROWS_PER_LANE					= 1 << LOG_ROWS_PER_LANE,

		// Padded stride between lanes (in partials)
		LANE_STRIDE						= (NO_PADDING) ?
											PARTIALS_PER_LANE :
											ROWS_PER_LANE * PADDED_PARTIALS_PER_ROW,

		// Number of elements needed to back this level of the raking grid
		RAKING_ELEMENTS					= ROWS * PADDED_PARTIALS_PER_ROW,
	};

	// If there are prefix dependences between lanes, a secondary raking grid
	// type will be needed in the event we have more than one warp of raking threads

	typedef typename If<_DEPENDENT_LANES && (LOG_RAKING_THREADS > LOG_WARP_THREADS),
		RakingGrid<										// Yes secondary grid
			T,													// Partial type
			LOG_RAKING_THREADS,									// Depositing threads (the primary raking threads)
			0,													// 1 lane (the primary raking threads only make one deposit)
			LOG_WARP_THREADS,									// Raking threads (1 warp)
			false>,												// There is only one lane, so there are no inter-lane prefix dependences
		NullType>										// No secondary grid
			::Type SecondaryGrid;


	/**
	 * Utility class for totaling the SMEM elements needed for an raking grid hierarchy
	 */
	template <typename RakingGrid, int __dummy = 0>
	struct TotalRakingElements
	{
		// Recurse
		enum { VALUE = RakingGrid::RAKING_ELEMENTS + TotalRakingElements<typename RakingGrid::SecondaryGrid>::VALUE };
	};
	template <int __dummy>
	struct TotalRakingElements<NullType, __dummy>
	{
		// Terminate
		enum { VALUE = 0 };
	};


	enum {
		// Total number of smem raking elements needed back this hierarchy
		// of raking grids (may be reused for other purposes)
		TOTAL_RAKING_ELEMENTS = TotalRakingElements<RakingGrid>::VALUE,
	};


	/**
	 * Type of pointer for inserting partials into lanes, e.g., lane_partial[LANE][0] = ...
	 */
	typedef T (*LanePartial)[LANE_STRIDE];


	/**
	 * Type of pointer for raking across lane segments
	 */
	typedef T* RakingSegment;


	/**
	 * Returns the location in the smem grid where the calling thread can insert/extract
	 * its partial for raking reduction/scan into the first lane.  Positions in subsequent
	 * lanes can be obtained via increments of LANE_STRIDE.
	 */
	static __host__ __device__ __forceinline__  LanePartial MyLanePartial(
		T *smem,
		int tid = threadIdx.x)
	{
/*
		int row = tid >> LOG_PARTIALS_PER_ROW;
		int col = tid & (PARTIALS_PER_ROW - 1);

		return (LanePartial) (smem + (row * PADDED_PARTIALS_PER_ROW) + col);
*/
		return (LanePartial) (smem + tid + (tid >> LOG_PARTIALS_PER_ROW));
	}


	/**
	 * Returns the location in the smem grid where the calling thread can begin serial
	 * raking/scanning
	 */
	static __host__ __device__ __forceinline__  RakingSegment MyRakingSegment(T *smem)
	{
/*
		int row = threadIdx.x >> LOG_SEGS_PER_ROW;
		int col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_PARTIALS_PER_SEG;
		return (RakingSegment) (smem + (row * PADDED_PARTIALS_PER_ROW) + col);
*/
		return (RakingSegment) (smem + (threadIdx.x << LOG_PARTIALS_PER_SEG) +
			(threadIdx.x >> LOG_SEGS_PER_ROW));
	}


	/**
	 * Displays configuration to standard out
	 */
	static __host__ __device__ __forceinline__ void Print()
	{
		printf("SCAN_LANES: %d\n"
				"PARTIALS_PER_LANE: %d\n"
				"RAKING_THREADS: %d\n"
				"RAKING_THREADS_PER_LANE: %d\n"
				"PARTIALS_PER_SEG: %d\n"
				"PARTIALS_PER_BANK_ARRAY: %d\n"
				"SEGS_PER_BANK_ARRAY: %d\n"
				"NO_PADDING: %d\n"
				"SEGS_PER_ROW: %d\n"
				"PARTIALS_PER_ROW: %d\n"
				"BANK_PADDING_PARTIALS: %d\n"
				"LANE_PADDING_PARTIALS: %d\n"
				"PADDED_PARTIALS_PER_ROW: %d\n"
				"ROWS: %d\n"
				"ROWS_PER_LANE: %d\n"
				"LANE_STRIDE: %d\n"
				"RAKING_ELEMENTS: %d\n",
			SCAN_LANES,
			PARTIALS_PER_LANE,
			RAKING_THREADS,
			RAKING_THREADS_PER_LANE,
			PARTIALS_PER_SEG,
			PARTIALS_PER_BANK_ARRAY,
			SEGS_PER_BANK_ARRAY,
			NO_PADDING,
			SEGS_PER_ROW,
			PARTIALS_PER_ROW,
			BANK_PADDING_PARTIALS,
			LANE_PADDING_PARTIALS,
			PADDED_PARTIALS_PER_ROW,
			ROWS,
			ROWS_PER_LANE,
			LANE_STRIDE,
			RAKING_ELEMENTS);
	}
};







} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
