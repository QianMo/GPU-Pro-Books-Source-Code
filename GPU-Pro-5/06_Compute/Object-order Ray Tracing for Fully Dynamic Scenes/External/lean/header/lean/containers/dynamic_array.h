/*****************************************************/
/* lean Containers              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONTAINERS_DYNAMIC_ARRAY
#define LEAN_CONTAINERS_DYNAMIC_ARRAY

#include "../lean.h"
#include "../memory/default_heap.h"
#include "../functional/variadic.h"
#include "../meta/type_traits.h"
#include "../tags/noncopyable.h"
#include "construction.h"

namespace lean 
{
namespace containers
{

/// Dynamic array base class.
template <class Element, class Heap>
struct dynamic_array_base : public noncopyable
{
	/// Type of the heap used by this vector.
	typedef Heap heap_type;
	/// Type of the size returned by this vector.
	typedef typename heap_type::size_type size_type;

	/// Type of the elements contained by this vector.
	typedef Element value_type;

	/// First element in the array.
	value_type *m_elements;
	/// One past the last element in the array.
	value_type *m_elementsEnd;

	/// Allocates the given number of elements.
	static LEAN_INLINE value_type* allocate(size_type capacity)
	{
		return (capacity > 0)
			? static_cast<value_type*>( heap_type::allocate(capacity * sizeof(value_type)) )
			: nullptr;
	}
	
	/// Frees all memory.
	LEAN_INLINE void free()
	{
		if (m_elements)
		{
			value_type *oldElements = m_elements;
			m_elements = nullptr;
			m_elementsEnd = nullptr;
			heap_type::free(oldElements);
		}
	}

	/// Constructs an empty vector.
	LEAN_INLINE dynamic_array_base()
		: m_elements(nullptr),
		m_elementsEnd(nullptr) { }
	/// Constructs an empty vector.
	LEAN_INLINE explicit dynamic_array_base(size_type capacity)
		: m_elements( allocate(capacity) ),
		m_elementsEnd( m_elements ) { }
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves all elements from the given vector to this vector.
	LEAN_INLINE dynamic_array_base(dynamic_array_base &&right) throw()
		: m_elements(right.m_elements),
		m_elementsEnd(right.m_elementsEnd)
	{
		right.m_elements = nullptr;
		right.m_elementsEnd = nullptr;
	}
#endif
	/// Moves all elements from the given vector to this vector.
	LEAN_INLINE dynamic_array_base(dynamic_array_base &right, consume_t) throw()
		: m_elements(right.m_elements),
		m_elementsEnd(right.m_elementsEnd)
	{
		right.m_elements = nullptr;
		right.m_elementsEnd = nullptr;
	}
	/// Releases all memory.
	LEAN_INLINE ~dynamic_array_base()
	{
		if (m_elements)
			heap_type::free(m_elements);
	}

	/// Moves all elements from the given vector to this vector.
	LEAN_INLINE void assign(dynamic_array_base &right, consume_t) throw()
	{
		if (&right != this)
		{
			free();

			m_elements = right.m_elements;
			m_elementsEnd = right.m_elementsEnd;

			right.m_elements = nullptr;
			right.m_elementsEnd = nullptr;
		}
	}

#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves all elements from the given vector to this vector.
	LEAN_INLINE void assign(dynamic_array_base &&right) throw()
	{
		assign(right, consume);
	}
	/// Moves all elements from the given vector to this vector.
	LEAN_INLINE dynamic_array_base& operator =(dynamic_array_base &&right) throw()
	{
		assign(right, consume);
		return *this;
	}
#endif

	/// Swaps the contents of this vector and the given vector.
	LEAN_INLINE void swap(dynamic_array_base &right) throw()
	{
		using std::swap;

		swap(m_elements, right.m_elements);
		swap(m_elementsEnd, right.m_elementsEnd);
	}
};

/// Dynamic array class.
template <class Element, class Heap = default_heap>
class dynamic_array : private dynamic_array_base<Element, Heap>
{
public:
	/// Type of the heap used by this vector.
	typedef Heap heap_type;
	/// Type of the size returned by this vector.
	typedef typename heap_type::size_type size_type;

	/// Type of the elements contained by this vector.
	typedef Element value_type;
	/// Type of pointers to the elements contained by this vector.
	typedef value_type* pointer;
	/// Type of constant pointers to the elements contained by this vector.
	typedef const value_type* const_pointer;
	/// Type of references to the elements contained by this vector.
	typedef value_type& reference;
	/// Type of constant references to the elements contained by this vector.
	typedef const value_type& const_reference;

	/// Type of iterators to the elements contained by this vector.
	typedef pointer iterator;
	/// Type of constant iterators to the elements contained by this vector.
	typedef const_pointer const_iterator;

private:
	typedef dynamic_array_base<Element, Heap> base_type;

	// Make sure size_type is unsigned
	LEAN_STATIC_ASSERT(is_unsigned<size_type>::value);

	/// Frees all elements.
	LEAN_INLINE void destroy()
	{
		if (m_elements)
		{
			clear();
			this->free();
		}
	}

public:
	/// Constructs an empty vector.
	dynamic_array() { }
	/// Constructs an empty vector.
	explicit dynamic_array(size_type capacity)
		: base_type(capacity) { }
	/// Copies all elements from the given vector to this vector.
	dynamic_array(const dynamic_array &right)
		: base_type(right.size())
	{
		m_elementsEnd = containers::copy_construct(right.m_elements, right.m_elementsEnd, m_elements, no_allocator);
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves all elements from the given vector to this vector.
	dynamic_array(dynamic_array &&right) noexcept
		: base_type(std::move(right)) { }
#endif
	/// Moves all elements from the given vector to this vector.
	dynamic_array(dynamic_array &right, consume_t)
		: base_type(right, consume) { }
	/// Copies all elements from the given range to this vector.
	template <class Iterator>
	dynamic_array(Iterator begin, Iterator end)
		: base_type(end - begin)
	{
		m_elementsEnd = containers::copy_construct(begin, end, m_elements, no_allocator);
	}
	/// Moves all elements from the given range to this vector.
	template <class Iterator>
	dynamic_array(Iterator begin, Iterator end, consume_t)
		: base_type(end - begin)
	{
		m_elementsEnd = containers::move_construct(begin, end, m_elements, no_allocator);
	}
	/// Destroys all elements in this vector.
	~dynamic_array()
	{
		containers::destruct(m_elements, m_elementsEnd, no_allocator);
	}

	/// Copies all elements of the given vector to this vector.
	void assign(const dynamic_array &right)
	{
		if (&right != this)
		{
			reset(right.size());
			m_elementsEnd = containers::copy_construct(right.m_elements, right.m_elementsEnd, m_elements, no_allocator);
		}
	}
	/// Moves all elements from the given vector to this vector.
	void assign(dynamic_array &right, consume_t)
	{
		if (&right != this)
		{
			clear();
			this->base_type::assign(right, consume);
		}
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves all elements from the given vector to this vector.
	LEAN_INLINE void assign(dynamic_array &&right)
	{
		assign(right, consume);
	}
#endif

	/// Copies all elements of the given vector to this vector.
	LEAN_INLINE dynamic_array& operator =(const dynamic_array &right)
	{
		assign(right);
		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves all elements from the given vector to this vector.
	LEAN_INLINE dynamic_array& operator =(dynamic_array &&right) noexcept
	{
		assign(right, consume);
		return *this;
	}
#endif
	
	/// Copies all elements from the given range to this vector.
	template <class Iterator>
	void assign_disjoint(Iterator begin, Iterator end)
	{
		reset(end - begin);
		m_elementsEnd = containers::copy_construct(begin, end, m_elements, no_allocator);
	}
	/// Moves all elements from the given range to this vector.
	template <class Iterator>
	void assign_disjoint(Iterator begin, Iterator end, consume_t)
	{
		reset(end - begin);
		m_elementsEnd = containers::move_construct(begin, end, m_elements, no_allocator);
	}

	/// Returns a pointer to the next non-constructed element.
	LEAN_INLINE void* allocate_back()
	{
		return m_elementsEnd;
	}
	/// Marks the next element as constructed.
	LEAN_INLINE reference shift_back(value_type *newElement)
	{
		LEAN_ASSERT(newElement == m_elementsEnd);

		return *m_elementsEnd++;
	}

#ifdef DOXYGEN_READ_THIS
	/// Constructs an element at the back, passing the given arguments.
	reference emplace_back(...);
#else
	#define LEAN_DYNAMIC_ARRAY_EMPLACE_METHOD_DECL \
		reference emplace_back
	#define LEAN_DYNAMIC_ARRAY_EMPLACE_METHOD_BODY(call) \
		{ \
			return shift_back( new (allocate_back()) value_type##call ); \
		}
	LEAN_VARIADIC_TEMPLATE(LEAN_FORWARD, LEAN_DYNAMIC_ARRAY_EMPLACE_METHOD_DECL, LEAN_NOTHING, LEAN_DYNAMIC_ARRAY_EMPLACE_METHOD_BODY)
#endif

	/// Appends a default-constructed element to this vector.
	LEAN_INLINE reference push_back()
	{
		containers::default_construct(m_elementsEnd, no_allocator);
		return *m_elementsEnd++;
	}
	/// Appends a default-constructed element to this vector.
	LEAN_INLINE pointer push_back_n(size_type count)
	{
		Element *firstElement = m_elementsEnd;
		containers::default_construct(firstElement, firstElement + count, no_allocator);
		m_elementsEnd += count;
		return firstElement;
	}
	/// Appends the given element to this vector.
	LEAN_INLINE reference push_back(const value_type &value)
	{
		containers::copy_construct(m_elementsEnd, value, no_allocator);
		return *m_elementsEnd++;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Appends the given element to this vector.
	LEAN_INLINE reference push_back(value_type &&value)
	{
		containers::move_construct(m_elementsEnd, value, no_allocator);
		return *m_elementsEnd++;
	}
#endif
	/// Removes the last element from this vector.
	LEAN_INLINE void pop_back()
	{
		LEAN_ASSERT(!empty());

		containers::destruct(--m_elementsEnd, no_allocator);
	}

	/// Clears all elements from this vector.
	LEAN_INLINE void clear()
	{
		Element *oldElementsEnd = m_elementsEnd;
		m_elementsEnd = m_elements;
		containers::destruct(m_elements, oldElementsEnd, no_allocator);
	}

	/// Reserves space for the given number of elements.
	LEAN_INLINE void reset(size_type newCapacity)
	{
		destroy();

		if (newCapacity > 0)
		{
			m_elements = this->allocate(newCapacity);
			m_elementsEnd = m_elements;
		}
	}
	
	/// Gets the first element in the vector, access violation on failure.
	LEAN_INLINE reference front(void) { LEAN_ASSERT(!empty()); return *m_elements; };
	/// Gets the first element in the vector, access violation on failure.
	LEAN_INLINE const_reference front(void) const { LEAN_ASSERT(!empty()); return *m_elements; };
	/// Gets the last element in the vector, access violation on failure.
	LEAN_INLINE reference back(void) { LEAN_ASSERT(!empty()); return m_elementsEnd[-1]; };
	/// Gets the last element in the vector, access violation on failure.
	LEAN_INLINE const_reference back(void) const { LEAN_ASSERT(!empty()); return m_elementsEnd[-1]; };

	/// Gets an element by position, access violation on failure.
	LEAN_INLINE reference operator [](size_type pos) { return m_elements[pos]; };
	/// Gets an element by position, access violation on failure.
	LEAN_INLINE const_reference operator [](size_type pos) const { return m_elements[pos]; };

	/// Returns an iterator to the first element contained by this vector.
	LEAN_INLINE iterator begin(void) { return m_elements; };
	/// Returns a constant iterator to the first element contained by this vector.
	LEAN_INLINE const_iterator begin(void) const { return m_elements; };
	/// Returns an iterator beyond the last element contained by this vector.
	LEAN_INLINE iterator end(void) { return m_elementsEnd; };
	/// Returns a constant iterator beyond the last element contained by this vector.
	LEAN_INLINE const_iterator end(void) const { return m_elementsEnd; };

	/// Returns true if the vector is empty.
	LEAN_INLINE bool empty(void) const { return (m_elements == m_elementsEnd); };
	/// Returns the number of elements contained by this vector.
	LEAN_INLINE size_type size(void) const { return m_elementsEnd - m_elements; };

	/// Swaps the contents of this vector and the given vector.
	LEAN_INLINE void swap(dynamic_array &right) noexcept
	{
		static_cast<base_type&>(*this).swap(right);
	}
};

/// Swaps the contents of the given vectors.
template <class Element, class Heap>
LEAN_INLINE void swap(dynamic_array<Element, Heap> &left, dynamic_array<Element, Heap> &right) noexcept
{
	left.swap(right);
}

} // namespace

using containers::dynamic_array;

} // namespace

#endif