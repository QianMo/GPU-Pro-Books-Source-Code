/*****************************************************/
/* lean Memory                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MEMORY_ALIGNED
#define LEAN_MEMORY_ALIGNED

#include "../lean.h"
#include "alignment.h"
#include "default_heap.h"

namespace lean
{
namespace memory
{
	/// Aligns derived classes according to the given alignment template argument.
	/** @remarks MSC adds padding to make the size of aligned structures a multiple of their alignment, make sure to specify
	  * this base class first to allow for empty base class optimization.
	  * @see lean::memory::heap_bound */
	template <size_t Alignment, class Heap = default_heap>
	class aligned : public stack_aligned<Alignment>
	{
		LEAN_STATIC_INTERFACE_BEHAVIOR(aligned)

	private:
#ifndef LEAN0X_NO_DELETE_METHODS
		/// Cannot be aligned properly, therefore disabled.
		LEAN_INLINE void* operator new[](size_t size) = delete;
		/// Cannot be aligned properly, therefore disabled.
		LEAN_INLINE void operator delete[](void *memory) = delete;
#else
		/// Cannot be aligned properly, therefore disabled.
		LEAN_INLINE void* operator new[](size_t size);
		/// Cannot be aligned properly, therefore disabled.
		LEAN_INLINE void operator delete[](void *memory);
#endif

	public:
		/// Allocates an aligned block of memory of the given size.
		LEAN_INLINE void* operator new(size_t size)
		{
			return Heap::allocate<Alignment>(size);
		}
		/// Frees the given block of memory.
		LEAN_INLINE void operator delete(void *memory)
		{
			Heap::free<Alignment>(memory);
		}
	};

} // namespace

using memory::aligned;

} // namespace

#endif