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
 * Simple caching allocator for device memory allocations. The allocator is
 * thread-safe and capable of managing device allocations on multiple GPUs.
 ******************************************************************************/

#pragma once

#include <math.h>
#include <set>
#include <map>

#include "../util/ns_umbrella.cuh"
#include "../util/spinlock.cuh"
#include "../util/error_utils.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {


/**
 * Simple caching allocator for device memory allocations. The allocator is
 * thread-safe and is capable of managing cached device allocations on multiple GPUs.
 *
 * Allocations are rounded up to and categorized by bin size.  Bin sizes progress
 * geometrically in accordance with the growth factor "bin_growth" provided during
 * construction.  Unused device allocations within a larger bin cache are not
 * reused for allocation requests that categorize to smaller bin sizes.
 *
 * Allocation requests below (bin_growth ^ min_bin) are rounded up to
 * (bin_growth ^ min_bin).
 *
 * Allocations above (bin_growth ^ max_bin) are not rounded up to the nearest
 * bin and are simply freed when they are deallocated instead of being returned
 * to a bin-cache.
 *
 * If the total storage of cached allocations on a given GPU will exceed
 * (max_cached_bytes), allocations for that GPU are simply freed when they are
 * deallocated instead of being returned to their bin-cache.
 *
 * For example, the default-constructed CachedAllocator is configured with:
 * 		bin_growth = 8
 * 		min_bin = 3
 * 		max_bin = 7
 * 		max_cached_bytes = (bin_growth ^ max_bin) * 3) - 1 = 6,291,455 bytes
 *
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes per GPU
 *
 */
struct CachedAllocator
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	typedef int GpuOrdinal;

	enum
	{
		INVALID_GPU_ORDINAL = -1,
	};


	/**
	 * Integer pow function for unsigned base and exponent
	 */
	static __forceinline__ unsigned int IntPow(
		unsigned int base,
		unsigned int exp)
	{
		unsigned int retval = 1;
		while (exp > 0)
		{
			if (exp & 1) {
				retval = retval * base;		// multiply the result by the current base
			}
			base = base * base;				// square the base
			exp = exp >> 1;					// divide the exponent in half
		}
		return retval;
	}


	/**
	 * Round up to the nearest power-of
	 */
	static __forceinline__ void NearestPowerOf(
		unsigned int &power,
		size_t &rounded_bytes,
		unsigned int base,
		size_t value)
	{
		power = 0;
		rounded_bytes = 1;

		while (rounded_bytes < value)
		{
			rounded_bytes *= base;
			power++;
		}
	}

	/**
	 * Descriptor for device memory allocations
	 */
	struct BlockDescriptor
	{
		GpuOrdinal			gpu;		// GPU ordinal
		void* 				d_ptr;		// Device pointer
		size_t				bytes;		// Size of allocation in bytes
		unsigned int		bin;		// Bin enumeration

		// Constructor
		BlockDescriptor(void *d_ptr, GpuOrdinal gpu) :
			d_ptr(d_ptr),
			bytes(0),
			bin(0),
			gpu(gpu) {}

		// Constructor
		BlockDescriptor(size_t bytes, unsigned int bin, GpuOrdinal gpu) :
			d_ptr(NULL),
			bytes(bytes),
			bin(bin),
			gpu(gpu) {}

		// Comparison functor for comparing device pointers
		static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b)
		{
			if (a.gpu < b.gpu) {
				return true;
			} else if (a.gpu > b.gpu) {
				return false;
			} else {
				return (a.d_ptr < b.d_ptr);
			}
		}

		// Comparison functor for comparing allocation sizes
		static bool SizeCompare(const BlockDescriptor &a, const BlockDescriptor &b)
		{
			if (a.gpu < b.gpu) {
				return true;
			} else if (a.gpu > b.gpu) {
				return false;
			} else {
				return (a.bytes < b.bytes);
			}
		}
	};

	// BlockDescriptor comparator function interface
	typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

	// Set type for cached blocks (ordered by size)
	typedef std::set<BlockDescriptor, Compare> CachedBlocks;

	// Set type for live blocks (ordered by ptr)
	typedef std::set<BlockDescriptor, Compare> BusyBlocks;

	// Map type of gpu ordinals to the number of cached bytes cached by each GPU
	typedef std::map<GpuOrdinal, size_t> GpuCachedBytes;


	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	Spinlock 		spin_lock;			// Spinlock for thread-safety

	CachedBlocks 	cached_blocks;		// Set of cached device allocations available for reuse
	BusyBlocks 		live_blocks;		// Set of live device allocations currently in use

	unsigned int 	bin_growth;			// Geometric growth factor for bin-sizes
	unsigned int 	min_bin;			// Minimum bin enumeration
	unsigned int 	max_bin;			// Maximum bin enumeration

	size_t			min_bin_bytes;		// Minimum bin size
	size_t 			max_bin_bytes;		// Maximum bin size
	size_t 			max_cached_bytes;	// Maximum aggregate cached bytes per GPU

	GpuCachedBytes	cached_bytes;		// Map of GPU ordinal to aggregate cached bytes on that GPU


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor.
	 */
	CachedAllocator(
		unsigned int bin_growth,		// Geometric growth factor for bin-sizes
		unsigned int min_bin,			// Minimum bin
		unsigned int max_bin,			// Maximum bin
		size_t max_cached_bytes) :		// Maximum aggregate cached bytes per GPU
			spin_lock(0),
			cached_blocks(BlockDescriptor::SizeCompare),
			live_blocks(BlockDescriptor::PtrCompare),
			bin_growth(bin_growth),
			min_bin(min_bin),
			max_bin(max_bin),
			min_bin_bytes(IntPow(bin_growth, min_bin)),
			max_bin_bytes(IntPow(bin_growth, max_bin)),
			max_cached_bytes(max_cached_bytes)
	{}


	/**
	 * Constructor.  Configured with:
	 * 		bin_growth = 8
	 * 		min_bin = 3
	 * 		max_bin = 7
	 * 		max_cached_bytes = (bin_growth ^ max_bin) * 3) - 1 = 6,291,455 bytes
	 *
	 * 	which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
	 * 	and sets a maximum of 6,291,455 cached bytes per GPU
	 */
	CachedAllocator() :
		spin_lock(0),
		cached_blocks(BlockDescriptor::SizeCompare),
		live_blocks(BlockDescriptor::PtrCompare),
		bin_growth(8),
		min_bin(3),
		max_bin(7),
		min_bin_bytes(IntPow(bin_growth, min_bin)),
		max_bin_bytes(IntPow(bin_growth, max_bin)),
		max_cached_bytes((max_bin_bytes * 3) - 1)
	{}


	/**
	 * Sets the limit on the number bytes this allocator is allowed to
	 * cache per GPU.
	 */
	void SetMaxCachedBytes(size_t max_cached_bytes)
	{
		// Lock
		Lock(&spin_lock);

		this->max_cached_bytes = max_cached_bytes;

		// Unlock
		Unlock(&spin_lock);
	}


	/**
	 * Provides a suitable allocation of device memory for the given size
	 * on the specified GPU
	 */
	cudaError_t Allocate(void** d_ptr, size_t bytes, GpuOrdinal gpu)
	{
		bool locked 					= false;
		GpuOrdinal entrypoint_gpu 		= INVALID_GPU_ORDINAL;
		cudaError_t error 				= cudaSuccess;

		// Round up to nearest bin size
		unsigned int bin;
		size_t bin_bytes;
		NearestPowerOf(bin, bin_bytes, bin_growth, bytes);
		if (bin < min_bin) {
			bin = min_bin;
			bin_bytes = min_bin_bytes;
		}

		// Check if bin is greater than our maximum bin
		if (bin > max_bin)
		{
			// Allocate the request exactly and give out-of-range bin
			bin = (unsigned int) -1;
			bin_bytes = bytes;
		}

		BlockDescriptor search_key(bin_bytes, bin, gpu);

		// Lock
		if (!locked) {
			Lock(&spin_lock);
			locked = true;
		}

		do {
			// Find a free block big enough within the same bin on the same GPU
			CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
			if ((block_itr != cached_blocks.end()) &&
				(block_itr->gpu == gpu) &&
				(block_itr->bin == search_key.bin))
			{
				// Reuse existing cache block.  Insert into live blocks.
				search_key = *block_itr;
				live_blocks.insert(search_key);

				// Remove from free blocks
				cached_blocks.erase(block_itr);
				cached_bytes[gpu] -= search_key.bytes;
			}
			else
			{
				// Need to allocate a new cache block. Unlock.
				if (locked) {
					Unlock(&spin_lock);
					locked = false;
				}

				// Set to specified GPU
				error = cudaGetDevice(&entrypoint_gpu);
				if (util::B40CPerror(error, "cudaGetDevice failed ", __FILE__, __LINE__)) break;
				error = cudaSetDevice(gpu);
				if (util::B40CPerror(error, "cudaSetDevice failed ", __FILE__, __LINE__)) break;

				// Allocate
				error = cudaMalloc(&search_key.d_ptr, search_key.bytes);
				if (util::B40CPerror(error, "cudaMalloc failed ", __FILE__, __LINE__)) break;

				// Lock
				if (!locked) {
					Lock(&spin_lock);
					locked = true;
				}

				// Insert into live blocks
				live_blocks.insert(search_key);
			}
		} while(0);

		// Unlock
		if (locked) {
			Unlock(&spin_lock);
			locked = false;
		}

		// Attempt to revert back to previous GPU if necessary
		if (entrypoint_gpu != INVALID_GPU_ORDINAL)
		{
			error = cudaSetDevice(entrypoint_gpu);
			util::B40CPerror(error, "cudaSetDevice failed ", __FILE__, __LINE__);
		}

		// Copy device pointer to output parameter
		*d_ptr = search_key.d_ptr;

		return error;
	}


	/**
	 * Provides a suitable allocation of device memory for the given size
	 * on the current GPU
	 */
	cudaError_t Allocate(void** d_ptr, size_t bytes)
	{
		GpuOrdinal current_gpu;
		cudaError_t error = cudaGetDevice(&current_gpu);
		if (util::B40CPerror(error, "cudaGetDevice failed ", __FILE__, __LINE__))
			return error;

		return Allocate(d_ptr, bytes, current_gpu);
	}


	/**
	 * Returns a live allocation of GPU memory on the specified GPU to
	 * the allocator
	 */
	cudaError_t Deallocate(void* d_ptr, GpuOrdinal gpu)
	{
		bool locked 					= false;
		GpuOrdinal entrypoint_gpu 		= INVALID_GPU_ORDINAL;
		cudaError_t error 				= cudaSuccess;

		BlockDescriptor search_key(d_ptr, gpu);

		// Lock
		if (!locked) {
			Lock(&spin_lock);
			locked = true;
		}

		do {
			// Find corresponding block descriptor
			BusyBlocks::iterator block_itr = live_blocks.find(search_key);
			if (block_itr == live_blocks.end())
			{
				// Cannot find pointer
				error = util::B40CPerror(cudaErrorUnknown, "Deallocate failed ", __FILE__, __LINE__);
				break;
			}
			else
			{
				// Remove from live blocks
				search_key = *block_itr;
				live_blocks.erase(block_itr);

				// Check if we should keep the returned allocation
				if ((search_key.bin <= max_bin) &&
					(cached_bytes[gpu] + search_key.bytes <= max_cached_bytes))
				{
					// Insert returned allocation into free blocks
					cached_blocks.insert(search_key);
					cached_bytes[gpu] += search_key.bytes;
				}
				else
				{
					// Free the returned allocation.  Unlock.
					if (locked) {
						Unlock(&spin_lock);
						locked = false;
					}

					// Set to specified GPU
					error = cudaGetDevice(&entrypoint_gpu);
					if (util::B40CPerror(error, "cudaGetDevice failed ", __FILE__, __LINE__)) break;
					error = cudaSetDevice(gpu);
					if (util::B40CPerror(error, "cudaSetDevice failed ", __FILE__, __LINE__)) break;

					// Free device memory
					error = cudaFree(d_ptr);
					if (util::B40CPerror(error, "cudaFree failed ", __FILE__, __LINE__)) break;
				}
			}
		} while (0);

		// Unlock
		if (locked) {
			Unlock(&spin_lock);
			locked = false;
		}

		// Attempt to revert back to entry-point GPU if necessary
		if (entrypoint_gpu != INVALID_GPU_ORDINAL)
		{
			error = cudaSetDevice(entrypoint_gpu);
			util::B40CPerror(error, "cudaSetDevice failed ", __FILE__, __LINE__);
		}

		return error;
	}


	/**
	 * Returns a live allocation of device memory on the current GPU to the
	 * allocator
	 */
	cudaError_t Deallocate(void* d_ptr)
	{
		GpuOrdinal current_gpu;
		cudaError_t error = cudaGetDevice(&current_gpu);
		if (util::B40CPerror(error, "cudaGetDevice failed ", __FILE__, __LINE__))
			return error;

		return Deallocate(d_ptr, current_gpu);
	}


	/**
	 * Frees all cached device allocations on all GPUs
	 */
	cudaError_t FreeAllCached()
	{
		cudaError_t error 				= cudaSuccess;
		bool locked 					= false;
		GpuOrdinal entrypoint_gpu 		= INVALID_GPU_ORDINAL;
		GpuOrdinal current_gpu			= INVALID_GPU_ORDINAL;

		// Lock
		if (!locked) {
			Lock(&spin_lock);
			locked = true;
		}

		while (!cached_blocks.empty())
		{
			// Get first block
			CachedBlocks::iterator begin = cached_blocks.begin();

			// Get entry-point GPU ordinal if necessary
			if (entrypoint_gpu == INVALID_GPU_ORDINAL)
			{
				error = cudaGetDevice(&entrypoint_gpu);
				if (util::B40CPerror(error, "cudaGetDevice failed ", __FILE__, __LINE__)) break;
			}

			// Set current GPU ordinal if necessary
			if (begin->gpu != current_gpu)
			{
				error = cudaSetDevice(begin->gpu);
				if (util::B40CPerror(error, "cudaSetDevice failed ", __FILE__, __LINE__)) break;
				current_gpu = begin->gpu;
			}

			// Free device memory
			error = cudaFree(begin->d_ptr);
			if (util::B40CPerror(error, "cudaGetDevice failed ", __FILE__, __LINE__)) break;

			// Reduce balance and erase entry
			cached_bytes[current_gpu] -= begin->bytes;
			cached_blocks.erase(begin);
		}

		// Unlock
		if (locked) {
			Unlock(&spin_lock);
			locked = false;
		}

		// Attempt to revert back to entry-point GPU if necessary
		if (entrypoint_gpu != INVALID_GPU_ORDINAL)
		{
			error = cudaSetDevice(entrypoint_gpu);
			util::B40CPerror(error, "cudaSetDevice failed ", __FILE__, __LINE__);
		}

		return error;
	}
};




} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
