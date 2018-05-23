//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr.  All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include "D3D11EffectsLite.h"

void* operator new(size_t size, D3DEffectsLite::Allocator &alloc);
void operator delete(void *bytes, D3DEffectsLite::Allocator &alloc);

namespace D3DEffectsLite
{

template <typename T>
void AllocatorDelete(Allocator &alloc, T *p)
{
	if (p)
	{
		p->~T();
		alloc.Free(p);
	}
}

template <typename T>
T* AllocatorNewMultiple(Allocator &alloc, size_t count)
{
	T *p = static_cast<T*>( alloc.Allocate( static_cast<UINT>(sizeof(T) * count) ) );
	
	size_t i = 0;

	try
	{
		for (; i < count; ++i)
			new(static_cast<void*>(p + i)) T();
	}
	catch (...)
	{
		for (size_t j = i; j-- > 0; )
			p[j].~T();

		alloc.Free(p);
		throw;
	}

	return p;
}

template <typename T>
void AllocatorDeleteMultiple(Allocator &alloc, T *p, size_t count)
{
	if (p)
	{
		for (size_t i = count; i-- > 0; )
			p[i].~T();
		alloc.Free(p);
	}
}

/// STL allocator heap adapter.
template <class Element>
class custom_allocator
{
private:
	Allocator *const alloc;

public:
	/// Value type.
	typedef typename Element value_type;

	/// Pointer type.
	typedef typename value_type* pointer;
	/// Reference type.
	typedef typename value_type& reference;
	/// Pointer type.
	typedef typename const value_type* const_pointer;
	/// Reference type.
	typedef typename const value_type& const_reference;

	/// Size type.
	typedef size_t size_type;
	/// Pointer difference type.
	typedef ptrdiff_t difference_type;

	/// Allows for the creation of differently-typed equivalent allocators.
	template <class Other>
	struct rebind
	{
		/// Equivalent allocator allocating elements of type Other.
		typedef custom_allocator<Other> other;
	};
	
	/// Default constructor.
	custom_allocator(Allocator *alloc) : alloc(alloc) { }
	/// Copy constructor.
	template <class Other>
	custom_allocator(const custom_allocator<Other> &right) : alloc(right.allocator()) { }
	/// Assignment operator.
	custom_allocator& operator=(const custom_allocator &right) { assert(alloc == right.alloc); return *this; } 
	/// Assignment operator.
	template <class Other>
	custom_allocator& operator=(const custom_allocator<Other> &right) { assert(alloc == right.alloc); return *this; } 

	/// Allocates the given number of elements.
	pointer allocate(size_type count)
	{
		return reinterpret_cast<pointer>( alloc->Allocate( static_cast<UINT>(count * sizeof(value_type)) ) );
	}
	/// Allocates the given amount of memory.
	pointer allocate(size_type count, const void *)
	{
		return allocate(count);
	}
	/// Allocates the given amount of memory.
	void deallocate(pointer ptr, size_type)
	{
		alloc->Free(ptr);
	}

	/// Constructs a new element from the given value at the given pointer.
	void construct(pointer ptr, const value_type& value)
	{
		new(reinterpret_cast<void*>(ptr)) Element(value);
	}
	/// Constructs a new element from the given value at the given pointer.
	template <class Other>
	void construct(pointer ptr, const Other& value)
	{
		new(reinterpret_cast<void*>(ptr)) Element(value);
	}
	/// Destructs an element at the given pointer.
	void destroy(pointer ptr)
	{
		ptr->~Element();
	}

	/// Gets the address of the given element.
	pointer address(reference value) const
	{
		return reinterpret_cast<pointer>( &reinterpret_cast<char&>(value) );
	}
	/// Gets the address of the given element.
	const_pointer address(const_reference value) const
	{
		return reinterpret_cast<const_pointer>( &reinterpret_cast<const char&>(value) );
	}

	/// Estimates the maximum number of elements that may be constructed.
	size_type max_size() const
	{
		size_type count = static_cast<UINT>(-1) / sizeof(Element);
		return count;
	}

	/// Gets the wrapped allocator object.
	Allocator* allocator() const { return alloc; }
};

#ifndef DOXYGEN_SKIP_THIS

/// STL allocator heap adapter.
template <>
class custom_allocator<void>
{
private:
	Allocator *const alloc;

public:
	/// Value type.
	typedef void value_type;

	/// Pointer type.
	typedef value_type* pointer;
	/// Pointer type.
	typedef const value_type* const_pointer;

	/// Size type.
	typedef size_t size_type;
	/// Pointer difference type.
	typedef ptrdiff_t difference_type;

	/// Allows for the creation of differently-typed equivalent allocators.
	template <class Other>
	struct rebind
	{
		/// Equivalent allocator allocating elements of type Other.
		typedef custom_allocator<Other> other;
	};
	
	/// Default constructor.
	custom_allocator(Allocator *alloc) : alloc(alloc) { }
	/// Copy constructor.
	template <class Other>
	custom_allocator(const custom_allocator<Other> &right) : alloc(right.alloc) { }
	/// Assignment operator.
	custom_allocator& operator=(const custom_allocator &right) { assert(alloc == right.alloc); return *this; } 
	/// Assignment operator.
	template <class Other>
	custom_allocator& operator=(const custom_allocator<Other> &right) { assert(alloc == right.alloc); return *this; }

	/// Gets the wrapped allocator object.
	Allocator* allocator() const { return alloc; }
};

#endif

/// Checks the given two allocators for equivalence.
template <class Element, class Other>
bool operator ==(const custom_allocator<Element> &left, const custom_allocator<Other> &right)
{
	assert(left.allocator() == right.allocator()); 
	return true;
}

/// Checks the given two allocators for inequivalence.
template <class Element, class Other>
bool operator !=(const custom_allocator<Element> &left, const custom_allocator<Other> &right)
{
	assert(left.allocator() == right.allocator()); 
	return false;
}

} // namespace
