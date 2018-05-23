/*****************************************************/
/* lean Memory                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MEMORY_HEAP_BOUND
#define LEAN_MEMORY_HEAP_BOUND

#include "../lean.h"
#include "default_heap.h"

namespace lean
{
namespace memory
{
	/// Always allocates space for derived classes on the same specified heap.
	/// @see lean::memory::aligned
	template <class Heap = default_heap>
	class heap_bound
	{
		LEAN_STATIC_INTERFACE_BEHAVIOR(heap_bound)

	public:
		/// Allocates a block of memory of the given size.
		LEAN_INLINE void* operator new(size_t size)
		{
			return Heap::allocate(size);
		}
		/// Frees the given block of memory.
		LEAN_INLINE void operator delete(void *memory)
		{
			Heap::free(memory);
		}
		/// Allocates a block of memory of the given size.
		LEAN_INLINE void* operator new[](size_t size)
		{
			return Heap::allocate(size);
		}
		/// Frees the given block of memory.
		LEAN_INLINE void operator delete[](void *memory)
		{
			Heap::free(memory);
		}
	};

} // namespace

using memory::heap_bound;

} // namespace

#endif