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
 * Common device intrinsics (potentially specialized by architecture)
 ******************************************************************************/

#pragma once

#include "../util/cuda_properties.cuh"
#include "../util/reduction/warp_reduce.cuh"
#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {



/**
 * SHR_ADD (shift-right then add)
 */
__device__ __forceinline__ unsigned int SHR_ADD(unsigned int x, unsigned int shift, unsigned int addend)
{
	unsigned int ret;
#if __CUDA_ARCH__ >= 200
	asm("vshr.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
		"=r"(ret) : "r"(x), "r"(shift), "r"(addend));
#else
	ret = (x >> shift) + addend;
#endif
	return ret;
}


/**
 * SHL_ADD (shift-left then add)
 */

__device__ __forceinline__ unsigned int SHL_ADD(unsigned int x, unsigned int shift, unsigned int addend)
{
	unsigned int ret;
#if __CUDA_ARCH__ >= 200
	asm("vshl.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
		"=r"(ret) : "r"(x), "r"(shift), "r"(addend));
#else
	ret = (x << shift) + addend;
#endif
	return ret;
}


#if __CUDA_ARCH__ >= 200
/**
 * BFE (bitfield extract).   Extracts a bit field from source and places the
 * zero or sign-extended result in extract
 */
__device__ __forceinline__ unsigned int BFE(unsigned int source, unsigned int bit_start, unsigned int num_bits)
{
	unsigned int bits;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "r"(bit_start), "r"(num_bits));
	return bits;
}
#endif

/**
 * BFE (bitfield extract).   Extracts a bit field from source and places the
 * zero or sign-extended result in extract
 */
template <typename T>
__device__ __forceinline__ T BFE(T source, unsigned int bit_start, unsigned int num_bits)
{
	const T MASK = (T(1) << num_bits) - 1;
	return (source >> bit_start) & MASK;
}













/**
 * BFI (bitfield insert).  Inserts the first num_bits of y into x starting at bit_start
 */
__device__ __forceinline__ void BFI(unsigned int &ret, unsigned int x, unsigned int y, unsigned int bit_start, unsigned int num_bits)
{
#if __CUDA_ARCH__ >= 200
	asm("bfi.b32 %0, %1, %2, %3, %4;" :
		"=r"(ret) : "r"(y), "r"(x), "r"(bit_start), "r"(num_bits));
#else
	// TODO
#endif
}


/**
 * IADD3
 */
__device__ __forceinline__ unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z)
{
#if __CUDA_ARCH__ >= 200
	asm("vadd.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(x) : "r"(x), "r"(y), "r"(z));
#else
	x = x + y + z;
#endif
	return x;
}



/**
 * PMT (byte permute).  Pick four arbitrary bytes from two 32-bit registers, and
 * reassemble them into a 32-bit destination register
 */
__device__ __forceinline__ int PRMT(unsigned int a, unsigned int b, unsigned int index)
{
	int ret;
	asm("prmt.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(index));
	return ret;
}

__device__ __forceinline__ void BAR(int count)
{
	asm volatile("bar.sync 1, %0;" : : "r"(count));
}


__device__ __forceinline__ float FMUL_RZ(float a, float b)
{
	float d;
	asm("mul.rz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
	return d;
}

__device__ __forceinline__ float FFMA_RZ(float a, float b, float c)
{
	float d;
	asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
	return d;
}



/**
 * Expands packed nibbles into packed bytes
 */
__device__ __forceinline__ void NibblesToBytes(
	unsigned int &int_byte0,
	unsigned int &int_byte1,
	unsigned int int_nibbles)
{
	unsigned int nib_shifted = int_nibbles >> 4;

	int_byte0 = PRMT(int_nibbles, nib_shifted, 0x5140);
	int_byte0 &= 0x0f0f0f0f;

	int_byte1 = PRMT(int_nibbles, nib_shifted, 0x7362);
	int_byte1 &= 0x0f0f0f0f;
}


/**
 * Expands packed nibbles into packed bytes
 */
__device__ __forceinline__ void BytesToHalves(
	unsigned int int_halves[2],
	unsigned int int_bytes)
{
	int_halves[0] = PRMT(int_bytes, 0, 0x4140);
	int_halves[1] = PRMT(int_bytes, 0, 0x4342);
}








/**
 * Terminates the calling thread
 */
__device__ __forceinline__ void ThreadExit() {
	asm("exit;");
}	


/**
 * Returns the warp lane ID of the calling thread
 */
__device__ __forceinline__ unsigned int LaneId()
{
	unsigned int ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret) );
	return ret;
}


/**
 * The best way to tally a warp-vote
 */
template <int LOG_ACTIVE_WARPS, int LOG_ACTIVE_THREADS>
__device__ __forceinline__ int CubTallyWarpVote(int predicate)
{
#if __CUDA_ARCH__ >= 200
	return __popc(__ballot(predicate));
#else
	const int ACTIVE_WARPS = 1 << LOG_ACTIVE_WARPS;
	const int ACTIVE_THREADS = 1 << LOG_ACTIVE_THREADS;

	__shared__ volatile int storage[ACTIVE_WARPS + 1][ACTIVE_THREADS];

	int tid = threadIdx.x & (CUB_WARP_THREADS(__CUB_CUDA_ARCH__) - 1);
	int wid = threadIdx.x >> CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__);

	return reduction::WarpReduce<LOG_ACTIVE_THREADS>::Invoke(predicate, storage[wid], tid);
#endif
}


/**
 * The best way to tally a warp-vote in the first warp
 */
template <int LOG_ACTIVE_THREADS>
__device__ __forceinline__ int CubTallyWarpVote(
	int predicate,
	volatile int storage[2][1 << LOG_ACTIVE_THREADS])
{
#if __CUDA_ARCH__ >= 200
	return __popc(__ballot(predicate));
#else
	return reduction::WarpReduce<LOG_ACTIVE_THREADS>::Invoke(
		predicate, 
		storage);
#endif
}


/**
 * The best way to warp-vote-all
 */
template <int LOG_ACTIVE_WARPS, int LOG_ACTIVE_THREADS>
__device__ __forceinline__ int CubWarpVoteAll(int predicate)
{
#if __CUDA_ARCH__ >= 120
	return __all(predicate);
#else 
	const int ACTIVE_THREADS = 1 << LOG_ACTIVE_THREADS;
	return (CubTallyWarpVote<LOG_ACTIVE_WARPS, LOG_ACTIVE_THREADS>(predicate) == ACTIVE_THREADS);
#endif
}


/**
 * The best way to warp-vote-all in the first warp
 */
template <int LOG_ACTIVE_THREADS>
__device__ __forceinline__ int CubWarpVoteAll(int predicate)
{
#if __CUDA_ARCH__ >= 120
	return __all(predicate);
#else
	const int ACTIVE_THREADS = 1 << LOG_ACTIVE_THREADS;
	__shared__ volatile int storage[2][ACTIVE_THREADS];

	return (CubTallyWarpVote<LOG_ACTIVE_THREADS>(predicate, storage) == ACTIVE_THREADS);
#endif
}

/**
 * The best way to warp-vote-any
 */
template <int LOG_ACTIVE_WARPS, int LOG_ACTIVE_THREADS>
__device__ __forceinline__ int WarpVoteAny(int predicate)
{
#if __CUDA_ARCH__ >= 120
	return __any(predicate);
#else
	return CubTallyWarpVote<LOG_ACTIVE_WARPS, LOG_ACTIVE_THREADS>(predicate);
#endif
}


/**
 * The best way to warp-vote-any in the first warp
 */
template <int LOG_ACTIVE_THREADS>
__device__ __forceinline__ int WarpVoteAny(int predicate)
{
#if __CUDA_ARCH__ >= 120
	return __any(predicate);
#else
	const int ACTIVE_THREADS = 1 << LOG_ACTIVE_THREADS;
	__shared__ volatile int storage[2][ACTIVE_THREADS];

	return CubTallyWarpVote<LOG_ACTIVE_THREADS>(predicate, storage);
#endif
}


/**
 * Wrapper for performing atomic operations on integers of type size_t 
 */
template <typename T, int SizeT = sizeof(T)>
struct AtomicInt;

template <typename T>
struct AtomicInt<T, 4>
{
	static __device__ __forceinline__ T Add(T* ptr, T val)
	{
		return atomicAdd((unsigned int *) ptr, (unsigned int) val);
	}
};

template <typename T>
struct AtomicInt<T, 8>
{
	static __device__ __forceinline__ T Add(T* ptr, T val)
	{
		return atomicAdd((unsigned long long int *) ptr, (unsigned long long int) val);
	}
};


} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
