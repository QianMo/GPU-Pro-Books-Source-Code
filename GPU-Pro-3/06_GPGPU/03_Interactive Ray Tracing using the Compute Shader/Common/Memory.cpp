/*
 * pbrt source code Copyright(c) 1998-2004 Matt Pharr and Greg Humphreys
 *
 * All Rights Reserved.
 * For educational use only; commercial use expressly forbidden.
 * NO WARRANTY, express or implied, for this software.
 * (See file License.txt for complete license)
 */

// --------------------------------------------------------------------	//
// This code was modified by the authors of the demo. The original		//
// PBRT code is available at https://github.com/mmp/pbrt-v2. Basically, //
// we removed all STL-based implementation and it was merged with		//
// our current framework.												//
// --------------------------------------------------------------------	//

// memory.cpp*
#include "Memory.h"

// Memory Allocation Functions
void *AllocAligned(size_t size) 
{
#ifdef WINDOWS
   return _aligned_malloc(size, 64);
#elif defined(LINUX)
	void* ptr;
	/* posix_memalign Returns 0 if succesfull, non-zero means an error */
	if(posix_memalign(&ptr, 64,size))
		return NULL;
	else
		return ptr;
#endif
}

void FreeAligned(void *ptr) 
{
#ifdef WINDOWS
       _aligned_free(ptr);
#elif defined(LINUX)
	free(ptr);
#endif
}