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
 * Radix sorting enactor
 ******************************************************************************/

#pragma once

#include "../radix_sort/problem_instance.cuh"
#include "../radix_sort/pass_policy.cuh"
#include "../util/error_utils.cuh"
#include "../util/spine.cuh"
#include "../util/cuda_properties.cuh"
#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {


/**
 * Radix sorting enactor class
 */
struct Enactor
{
	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	// Temporary device storage needed for reducing partials produced
	// by separate CTAs
	util::Spine spine;

	// Device properties
	const util::CudaProperties cuda_props;


	//---------------------------------------------------------------------
	// Helper structures
	//---------------------------------------------------------------------

	/**
	 * Tuned pass policy whose type signature does not reflect the tuned
	 * SM architecture.
	 */
	template <
		typename 		ProblemInstance,
		ProblemSize 	PROBLEM_SIZE,
		int 			RADIX_BITS>
	struct OpaquePassPolicy
	{
		// The appropriate tuning arch-id from the arch-id targeted by the
		// active compiler pass.
		enum
		{
			COMPILER_TUNE_ARCH 		= (__CUB_CUDA_ARCH__ >= 200) ?
										200 :
										(__CUB_CUDA_ARCH__ >= 130) ?
											130 :
											100
		};

		// Tuned pass policy
		typedef TunedPassPolicy<
			COMPILER_TUNE_ARCH,
			ProblemInstance,
			PROBLEM_SIZE,
			RADIX_BITS> TunedPassPolicy;

		struct DispatchPolicy 	: TunedPassPolicy::DispatchPolicy {};
		struct UpsweepPolicy 	: TunedPassPolicy::UpsweepPolicy {};
		struct SpinePolicy 		: TunedPassPolicy::SpinePolicy {};
		struct DownsweepPolicy 	: TunedPassPolicy::DownsweepPolicy {};
		struct SinglePolicy 	: TunedPassPolicy::SinglePolicy {};
	};


	/**
	 * Helper structure for iterating passes.
	 */
	template <
		int 			TUNE_ARCH,
		typename 		ProblemInstance,
		ProblemSize 	PROBLEM_SIZE,
		int 			BITS_REMAINING,
		int 			CURRENT_BIT>
	struct IteratePasses
	{

		/**
		 * DispatchPass pass
		 */
		static cudaError_t DispatchPass(
			ProblemInstance &problem_instance,
			Enactor &enactor)
		{
			cudaError_t error = cudaSuccess;
			do {

				int sm_version = enactor.cuda_props.device_sm_version;
				int sm_count = enactor.cuda_props.device_props.multiProcessorCount;

				enum {
					PREFERRED_BITS = PreferredDigitBits<TUNE_ARCH>::PREFERRED_BITS,
				};

				// Tuned pass policy for preferred bits
				typedef radix_sort::TunedPassPolicy<TUNE_ARCH, ProblemInstance, PROBLEM_SIZE, PREFERRED_BITS> TunedPassPolicy;

				if (problem_instance.num_elements <= TunedPassPolicy::SinglePolicy::TILE_ELEMENTS)
				{
					// Single CTA pass
					typedef OpaquePassPolicy<ProblemInstance, PROBLEM_SIZE, PREFERRED_BITS> OpaquePassPolicy;

					// Print debug info
					if (problem_instance.debug)
					{
						printf("\nCurrent bit(%d), Radix bits(%d), tuned arch(%d), SM arch(%d)\n",
							CURRENT_BIT, PREFERRED_BITS, TUNE_ARCH, enactor.cuda_props.device_sm_version);
						fflush(stdout);
					}

					// Single kernel props
					typename ProblemInstance::SingleKernelProps single_props;
					error = single_props.template Init<
						typename TunedPassPolicy::SinglePolicy,
						typename OpaquePassPolicy::SinglePolicy>(sm_version, sm_count);
					if (error) break;

					// Dispatch current pass
					error = problem_instance.DispatchPass(
						CURRENT_BIT,
						BITS_REMAINING,
						single_props);
					if (error) break;

				}
				else
				{
					// Multi-CTA pass
					enum {
						RADIX_BITS = CUB_MIN(BITS_REMAINING, (BITS_REMAINING % PREFERRED_BITS == 0) ? PREFERRED_BITS : PREFERRED_BITS - 1),
					};

					// Print debug info
					if (problem_instance.debug)
					{
						printf("\nCurrent bit(%d), Radix bits(%d), tuned arch(%d), SM arch(%d)\n",
							CURRENT_BIT, RADIX_BITS, TUNE_ARCH, enactor.cuda_props.device_sm_version);
						fflush(stdout);
					}

					// Redefine tuned and opaque pass policies
					typedef radix_sort::TunedPassPolicy<TUNE_ARCH, ProblemInstance, PROBLEM_SIZE, RADIX_BITS> 	TunedPassPolicy;
					typedef OpaquePassPolicy<ProblemInstance, PROBLEM_SIZE, RADIX_BITS>							OpaquePassPolicy;

					// Upsweep kernel props
					typename ProblemInstance::UpsweepKernelProps upsweep_props;
					error = upsweep_props.template Init<
						typename TunedPassPolicy::UpsweepPolicy,
						typename OpaquePassPolicy::UpsweepPolicy>(sm_version, sm_count);
					if (error) break;

					// Spine kernel props
					typename ProblemInstance::SpineKernelProps spine_props;
					error = spine_props.template Init<
						typename TunedPassPolicy::SpinePolicy,
						typename OpaquePassPolicy::SpinePolicy>(sm_version, sm_count);
					if (error) break;

					// Downsweep kernel props
					typename ProblemInstance::DownsweepKernelProps downsweep_props;
					error = downsweep_props.template Init<
						typename TunedPassPolicy::DownsweepPolicy,
						typename OpaquePassPolicy::DownsweepPolicy>(sm_version, sm_count);
					if (error) break;

					// Dispatch current pass
					error = problem_instance.DispatchPass(
						RADIX_BITS,
						CURRENT_BIT,
						upsweep_props,
						spine_props,
						downsweep_props,
						TunedPassPolicy::DispatchPolicy::UNIFORM_GRID_SIZE,
						TunedPassPolicy::DispatchPolicy::DYNAMIC_SMEM_CONFIG);
					if (error) break;

					// DispatchPass next pass
					error = IteratePasses<
						TUNE_ARCH,
						ProblemInstance,
						PROBLEM_SIZE,
						BITS_REMAINING - RADIX_BITS,
						CURRENT_BIT + RADIX_BITS>::DispatchPass(problem_instance, enactor);
					if (error) break;
				}

			} while (0);

			return error;
		}
	};


	/**
	 * Helper structure for iterating passes. (Termination)
	 */
	template <
		int TUNE_ARCH,
		typename ProblemInstance,
		ProblemSize PROBLEM_SIZE,
		int CURRENT_BIT>
	struct IteratePasses<TUNE_ARCH, ProblemInstance, PROBLEM_SIZE, 0, CURRENT_BIT>
	{
		/**
		 * DispatchPass pass
		 */
		static cudaError_t DispatchPass(
			ProblemInstance &problem_instance,
			Enactor &enactor)
		{
			return cudaSuccess;
		}
	};


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	Enactor() {}


	/**
	 * Enact a sort.
	 *
	 * @param problem_storage
	 * 		Instance of b40c::util::DoubleBuffer
	 * @param num_elements
	 * 		The number of elements in problem_storage to sort (starting at offset 0)
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		ProblemSize PROBLEM_SIZE,
		int BITS_REMAINING,
		int CURRENT_BIT,
		typename DoubleBuffer>
	cudaError_t Sort(
		DoubleBuffer& 	problem_storage,
		int 			num_elements,
		cudaStream_t	stream 			= 0,
		int 			max_grid_size 	= 0,
		bool 			debug 			= false)
	{
		typedef ProblemInstance<DoubleBuffer, int> ProblemInstance;

		if (num_elements <= 1) {
			// Nothing to do
			return cudaSuccess;
		}

		ProblemInstance problem_instance(
			problem_storage,
			num_elements,
			stream,
			spine,
			max_grid_size,
			debug);

		if (cuda_props.kernel_ptx_version >= 200)
		{
			return IteratePasses<200, ProblemInstance, PROBLEM_SIZE, BITS_REMAINING, CURRENT_BIT>::DispatchPass(problem_instance, *this);
		}
		else if (cuda_props.kernel_ptx_version >= 130)
		{
			return IteratePasses<130, ProblemInstance, PROBLEM_SIZE, BITS_REMAINING, CURRENT_BIT>::DispatchPass(problem_instance, *this);
		}
		else
		{
			return IteratePasses<100, ProblemInstance, PROBLEM_SIZE, BITS_REMAINING, CURRENT_BIT>::DispatchPass(problem_instance, *this);
		}
	}


	/**
	 * Enact a sort.
	 *
	 * @param problem_storage
	 * 		Instance of b40c::util::DoubleBuffer
	 * @param num_elements
	 * 		The number of elements in problem_storage to sort (starting at offset 0)
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename DoubleBuffer>
	cudaError_t Sort(
		DoubleBuffer& 	problem_storage,
		int 			num_elements,
		cudaStream_t	stream 			= 0,
		int 			max_grid_size 	= 0,
		bool 			debug 			= false)
	{
		return Sort<
			LARGE_PROBLEM,
			sizeof(typename DoubleBuffer::KeyType) * 8,			// BITS_REMAINING
			0>(													// CURRENT_BIT
				problem_storage,
				num_elements,
				stream,
				max_grid_size,
				debug);
	}


	/**
	 * Enact a sort on a small problem (0 < n < 100,000 elements)
	 *
	 * @param problem_storage
	 * 		Instance of b40c::util::DoubleBuffer
	 * @param num_elements
	 * 		The number of elements in problem_storage to sort (starting at offset 0)
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename DoubleBuffer>
	cudaError_t SmallSort(
		DoubleBuffer& 	problem_storage,
		int 			num_elements,
		cudaStream_t	stream 			= 0,
		int 			max_grid_size 	= 0,
		bool 			debug 			= false)
	{
		return Sort<
			SMALL_PROBLEM,
			sizeof(typename DoubleBuffer::KeyType) * 8,			// BITS_REMAINING
			0>(													// CURRENT_BIT
				problem_storage,
				num_elements,
				stream,
				max_grid_size,
				debug);
	}
};





} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
