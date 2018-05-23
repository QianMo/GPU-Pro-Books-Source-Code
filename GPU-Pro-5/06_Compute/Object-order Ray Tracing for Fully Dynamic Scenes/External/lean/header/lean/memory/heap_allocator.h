/*****************************************************/
/* lean Memory                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MEMORY_HEAP_ALLOCATOR
#define LEAN_MEMORY_HEAP_ALLOCATOR

#include "../lean.h"
#include "../meta/strip.h"
#include "alignment.h"
#include "default_heap.h"

namespace lean
{
namespace memory
{

/// STL allocator heap adapter.
template <class Element, class Heap = default_heap, size_t AlignmentOrZero = 0>
class heap_allocator
{
public:
	/// Alignment.
	struct alignment
	{
		/// Alignment.
		static const size_t value = (AlignmentOrZero) ? AlignmentOrZero : alignof(Element);
	};

	/// Heap adapted by this heap allocator.
	typedef Heap heap_type;

	/// Value type.
	typedef typename strip_const<Element>::type value_type;

	/// Pointer type.
	typedef typename value_type* pointer;
	/// Reference type.
	typedef typename value_type& reference;
	/// Pointer type.
	typedef typename const value_type* const_pointer;
	/// Reference type.
	typedef typename const value_type& const_reference;

	/// Size type.
	typedef typename heap_type::size_type size_type;
	/// Pointer difference type.
	typedef ptrdiff_t difference_type;

	/// Allows for the creation of differently-typed equivalent allocators.
	template <class Other>
	struct rebind
	{
		/// Equivalent allocator allocating elements of type Other.
		typedef heap_allocator<Other, Heap, AlignmentOrZero> other;
	};
	
	/// Default constructor.
	LEAN_INLINE heap_allocator() { }
	/// Copy constructor.
	template <class Other>
	LEAN_INLINE heap_allocator(const heap_allocator<Other, Heap, AlignmentOrZero> &right) { }
	/// Assignment operator.
	template <class Other>
	LEAN_INLINE heap_allocator& operator=(const heap_allocator<Other, Heap, AlignmentOrZero> &right) { return *this; }
	
	/// Allocates the given number of elements.
	LEAN_INLINE pointer allocate(size_type count)
	{
		return reinterpret_cast<pointer>( heap_type::allocate<alignment::value>(count * sizeof(value_type)) );
	}
	/// Allocates the given amount of memory.
	LEAN_INLINE pointer allocate(size_type count, const void *)
	{
		return allocate(count);
	}
	/// Allocates the given amount of memory.
	LEAN_INLINE void deallocate(pointer ptr, size_type)
	{
		heap_type::free<alignment::value>(ptr);
	}

	/// Constructs a new element from the given value at the given pointer.
	LEAN_INLINE void construct(pointer ptr, const value_type& value)
	{
		new(reinterpret_cast<void*>(ptr)) Element(value);
	}
	/// Constructs a new element from the given value at the given pointer.
	template <class Other>
	LEAN_INLINE void construct(pointer ptr, const Other& value)
	{
		new(reinterpret_cast<void*>(ptr)) Element(value);
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Constructs a new element from the given value at the given pointer.
	LEAN_INLINE void construct(pointer ptr, value_type&& value)
	{
		new(reinterpret_cast<void*>(ptr)) Element(std::move(value));
	}
	/// Constructs a new element from the given value at the given pointer.
	template <class Other>
	LEAN_INLINE void construct(pointer ptr, Other&& value)
	{
		new(reinterpret_cast<void*>(ptr)) Element(std::forward<Other>(value));
	}
#endif
	/// Destructs an element at the given pointer.
	LEAN_INLINE void destroy(pointer ptr)
	{
		ptr->~Element();
	}

	/// Gets the address of the given element.
	LEAN_INLINE pointer address(reference value) const
	{
		return reinterpret_cast<pointer>( &reinterpret_cast<char&>(value) );
	}
	/// Gets the address of the given element.
	LEAN_INLINE const_pointer address(const_reference value) const
	{
		return reinterpret_cast<const_pointer>( &reinterpret_cast<const char&>(value) );
	}

	/// Estimates the maximum number of elements that may be constructed.
	LEAN_INLINE size_type max_size() const
	{
		size_type count = static_cast<size_type>(-1) / sizeof(Element);
		return (0 < count) ? count : 1;
	}
};

#ifndef DOXYGEN_SKIP_THIS

/// STL allocator heap adapter.
template <class Heap, size_t AlignmentOrZero>
class heap_allocator<void, Heap, AlignmentOrZero>
{
public:
	/// Heap adapted by this heap allocator.
	typedef Heap heap_type;

	/// Value type.
	typedef void value_type;

	/// Pointer type.
	typedef value_type* pointer;
	/// Pointer type.
	typedef const value_type* const_pointer;

	/// Size type.
	typedef typename heap_type::size_type size_type;
	/// Pointer difference type.
	typedef ptrdiff_t difference_type;

	/// Allows for the creation of differently-typed equivalent allocators.
	template <class Other>
	struct rebind
	{
		/// Equivalent allocator allocating elements of type Other.
		typedef heap_allocator<Other, Heap, AlignmentOrZero> other;
	};
	
	/// Default constructor.
	LEAN_INLINE heap_allocator() { }
	/// Copy constructor.
	template <class Other>
	LEAN_INLINE heap_allocator(const heap_allocator<Other, Heap, AlignmentOrZero> &right) { }
	/// Assignment operator.
	template <class Other>
	LEAN_INLINE heap_allocator& operator=(const heap_allocator<Other, Heap, AlignmentOrZero> &right) { return *this; }
};

#endif

/// Checks the given two allocators for equivalence.
template <class Element, class Heap, size_t AlignmentOrZero, class Other>
LEAN_INLINE bool operator ==(const heap_allocator<Element, Heap, AlignmentOrZero>&, const heap_allocator<Other, Heap, AlignmentOrZero>&)
{
	return true;
}

/// Checks the given two allocators for inequivalence.
template <class Element, class Heap, size_t AlignmentOrZero, class Other>
LEAN_INLINE bool operator !=(const heap_allocator<Element, Heap, AlignmentOrZero>&, const heap_allocator<Other, Heap, AlignmentOrZero>&)
{
	return false;
}

} // namespace

using memory::heap_allocator;

} // namespace

#endif