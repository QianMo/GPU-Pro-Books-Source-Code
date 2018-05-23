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
 * Management of temporary device storage needed for maintaining partial
 * reductions between subsequent grids
 ******************************************************************************/

#pragma once

#include "../util/error_utils.cuh"
#include "../util/cuda_properties.cuh"
#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {

/**
 * Manages device storage needed for communicating partial reductions
 * between CTAs in subsequent grids
 */
struct Spine
{
	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Device spine storage
	void *d_spine;

	// Host-mapped spine storage (if so constructed)
	void *h_spine;

	// Number of bytes backed by d_spine
	size_t spine_bytes;

	// GPU d_spine was allocated on
	int gpu;

	// Whether or not the spine has a shadow spine on the host
	bool host_shadow;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor (device-allocated spine)
	 */
	Spine() :
		d_spine(NULL),
		h_spine(NULL),
		spine_bytes(0),
		gpu(CUB_INVALID_DEVICE),
		host_shadow(false) {}


	/**
	 * Constructor
	 *
	 * @param host_shadow
	 * 		Whether or not the spine has a shadow spine on the host
	 */
	Spine(bool host_shadow) :
		d_spine(NULL),
		h_spine(NULL),
		spine_bytes(0),
		gpu(CUB_INVALID_DEVICE),
		host_shadow(host_shadow) {}


	/**
	 * Deallocates and resets the spine
	 */
	cudaError_t HostReset()
	{
		cudaError_t retval = cudaSuccess;
		do {

			if (gpu == CUB_INVALID_DEVICE) return retval;

			// Save current gpu
			int current_gpu;
			if (retval = util::B40CPerror(cudaGetDevice(&current_gpu),
				"Spine cudaGetDevice failed: ", __FILE__, __LINE__)) break;
#if CUDA_VERSION >= 4000
			if (retval = util::B40CPerror(cudaSetDevice(gpu),
				"Spine cudaSetDevice failed: ", __FILE__, __LINE__)) break;
#endif
			if (d_spine) {
				// Deallocate
				if (retval = util::B40CPerror(cudaFree(d_spine),
					"Spine cudaFree d_spine failed: ", __FILE__, __LINE__)) break;
				d_spine = NULL;
			}
			if (h_spine) {
				// Deallocate
				if (retval = util::B40CPerror(cudaFreeHost((void *) h_spine),
					"Spine cudaFreeHost h_spine failed", __FILE__, __LINE__)) break;

				h_spine = NULL;
			}

#if CUDA_VERSION >= 4000
			// Restore current gpu
			if (retval = util::B40CPerror(cudaSetDevice(current_gpu),
				"Spine cudaSetDevice failed: ", __FILE__, __LINE__)) break;
#endif

			gpu 			= CUB_INVALID_DEVICE;
			spine_bytes	 	= 0;

		} while (0);

		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~Spine()
	{
		HostReset();
	}


	/**
	 * Device spine storage accessor
	 */
	void* operator()()
	{
		return d_spine;
	}


	/**
	 * Sets up the spine to accommodate the specified number of bytes.
	 * Reallocates if necessary.
	 */
	template <typename SizeT>
	cudaError_t Setup(SizeT problem_spine_bytes)
	{
		cudaError_t retval = cudaSuccess;
		do {
			// Get current gpu
			int current_gpu;
			if (retval = util::B40CPerror(cudaGetDevice(&current_gpu),
				"Spine cudaGetDevice failed: ", __FILE__, __LINE__)) break;

			// Check if big enough and if lives on proper GPU
			if ((problem_spine_bytes > spine_bytes) || (gpu != current_gpu)) {

				// Deallocate if exists
				if (retval = HostReset()) break;

				// Remember device
				gpu = current_gpu;

				// Reallocate
				spine_bytes = problem_spine_bytes;

				// Allocate on device
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_spine, spine_bytes),
					"Spine cudaMalloc d_spine failed", __FILE__, __LINE__)) break;

				if (host_shadow) {
					// Allocate pinned memory for h_spine
					int flags = cudaHostAllocMapped;
					if (retval = util::B40CPerror(cudaHostAlloc((void **)&h_spine, problem_spine_bytes, flags),
						"Spine cudaHostAlloc h_spine failed", __FILE__, __LINE__)) break;
				}
			}
		} while (0);

		return retval;
	}


	/**
	 * Syncs the shadow host spine with device spine
	 */
	cudaError_t Sync(cudaStream_t stream)
	{
		return cudaMemcpyAsync(
			h_spine,
			d_spine,
			spine_bytes,
			cudaMemcpyDeviceToHost,
			stream);
	}


	/**
	 * Syncs the shadow host spine with device spine
	 */
	cudaError_t Sync()
	{
		return cudaMemcpy(
			h_spine,
			d_spine,
			spine_bytes,
			cudaMemcpyDeviceToHost);
	}


};

} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
