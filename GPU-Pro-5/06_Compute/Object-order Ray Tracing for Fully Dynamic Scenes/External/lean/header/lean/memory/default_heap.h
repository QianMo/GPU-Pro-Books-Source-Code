/*****************************************************/
/* lean Memory                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MEMORY_DEFAULT_HEAP
#define LEAN_MEMORY_DEFAULT_HEAP

#ifndef LEAN_DEFAULT_HEAP

	#include "crt_heap.h"
	/// Default heap to be used by all subsequent definitions that make use of the heap concept.
	/// @ingroup MemorySwitches
	#define LEAN_DEFAULT_HEAP crt_heap

#endif

namespace lean
{
	namespace memory
	{
		/// Default heap to be used by all subsequent definitions that make use of the heap concept.
		typedef LEAN_DEFAULT_HEAP default_heap;

	} // namespace

	using memory::default_heap;

} // namespace

#ifdef DOXYGEN_READ_THIS
	/// Define this to override new using a custom specified default heap.
	/// @ingroup MemorySwitches
	#define LEAN_OVERRIDE_NEW
	#undef LEAN_OVERRIDE_NEW
#endif

#ifdef LEAN_OVERRIDE_NEW

	#if LEAN_DEFAULT_HEAP == crt_heap
		// CRT uses global operators new / delete, overriding them would cause chaos
		#error Cannot override new using the CRT heap
	#else
		/// Allocates memory using the previously defined default_heap.
		inline void* operator new(size_t size)
		{
			return lean::memory::default_heap::allocate(size);
		}
		/// Frees memory using the previously defined default_heap heap.
		inline void operator delete(void *memory)
		{
			lean::memory::default_heap::free(memory);
		}
		/// Allocates memory using the previously defined default_heap.
		inline void* operator new[](size_t size)
		{
			return lean::memory::default_heap::allocate(size);
		}
		/// Frees memory using the previously defined default_heap heap.
		inline void operator delete[](void *memory)
		{
			lean::memory::default_heap::free(memory);
		}
	#endif

#endif

#endif