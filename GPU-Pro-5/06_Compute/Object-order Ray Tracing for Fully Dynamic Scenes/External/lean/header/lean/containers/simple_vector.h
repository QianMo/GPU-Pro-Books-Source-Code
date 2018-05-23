/*****************************************************/
/* lean Containers              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONTAINERS_SIMPLE_VECTOR
#define LEAN_CONTAINERS_SIMPLE_VECTOR

#include "../lean.h"
#include "vector_policies.h"
#include "construction.h"
#include "allocator_aware.h"
#include "../meta/type_traits.h"
#include "../memory/heap_allocator.h"
#include <memory>
#include <stdexcept>

namespace lean 
{
namespace containers
{

/// Defines construction policies for the class simple_vector.
namespace simple_vector_policies = vector_policies;

/// Simple and fast vector class, partially implementing the STL vector interface.
template < class Element, class Policy = simple_vector_policies::nonpod, class Allocator = heap_allocator<Element> >
class simple_vector : protected allocator_aware_base<typename Allocator::template rebind<Element>::other>
{
private:
	typedef allocator_aware_base<typename Allocator::template rebind<Element>::other> base_type;

public:
	/// Construction policy used.
	typedef Policy construction_policy;

	/// Type of the allocator used by this vector.
	typedef typename base_type::allocator_type allocator_type;
	/// Type of the size returned by this vector.
	typedef typename allocator_type::size_type size_type;
	/// Type of the difference between the addresses of two elements in this vector.
	typedef typename allocator_type::difference_type difference_type;

	/// Type of pointers to the elements contained by this vector.
	typedef typename allocator_type::pointer pointer;
	/// Type of constant pointers to the elements contained by this vector.
	typedef typename allocator_type::const_pointer const_pointer;
	/// Type of references to the elements contained by this vector.
	typedef typename allocator_type::reference reference;
	/// Type of constant references to the elements contained by this vector.
	typedef typename allocator_type::const_reference const_reference;
	/// Type of the elements contained by this vector.
	typedef typename allocator_type::value_type value_type;

	/// Type of iterators to the elements contained by this vector.
	typedef pointer iterator;
	/// Type of constant iterators to the elements contained by this vector.
	typedef const_pointer const_iterator;

private:
	Element *m_elements;
	Element *m_elementsEnd;
	Element *m_capacityEnd;

	// Make sure size_type is unsigned
	LEAN_STATIC_ASSERT(is_unsigned<size_type>::value);

	LEAN_INLINE void default_construct(Element *dest)
	{
		if (!Policy::no_init)
		{
			base_type::allocator_ref allocRef(*this);
			containers::default_construct(dest, allocRef.allocator, typename Policy::construct_tag());
		}
	}
	LEAN_INLINE void default_construct(Element *dest, Element *destEnd)
	{
		if (!Policy::no_init)
		{
			base_type::allocator_ref allocRef(*this);
			containers::default_construct(dest, destEnd, allocRef.allocator, typename Policy::construct_tag());
		}
	}
	LEAN_INLINE void copy_construct(Element *dest, const Element &source)
	{
		base_type::allocator_ref allocRef(*this);
		containers::copy_construct(dest, source, allocRef.allocator, typename Policy::copy_tag());
	}
	template <class Iterator>
	LEAN_INLINE void copy_construct(Iterator source, Iterator sourceEnd, Element *dest)
	{
		base_type::allocator_ref allocRef(*this);
		containers::copy_construct(source, sourceEnd, dest, allocRef.allocator, typename Policy::copy_tag());
	}
	LEAN_INLINE void move_construct(Element *dest, Element &source)
	{
		base_type::allocator_ref allocRef(*this);
		// NOTE: Use copy tag, move tag only when no destruction takes place
		containers::move_construct(dest, source, allocRef.allocator, typename Policy::copy_tag());
	}
	template <class Iterator>
	LEAN_INLINE void move_construct(Iterator source, Iterator sourceEnd, Element *dest)
	{
		base_type::allocator_ref allocRef(*this);
		// NOTE: Use copy tag, move tag only when no destruction takes place
		containers::move_construct(source, sourceEnd, dest, allocRef.allocator, typename Policy::copy_tag());
	}
	LEAN_INLINE void destruct(Element *destr)
	{
		base_type::allocator_ref allocRef(*this);
		containers::destruct(destr, allocRef.allocator, typename Policy::destruct_tag());
	}
	LEAN_INLINE void destruct(Element *destr, Element *destrEnd)
	{
		base_type::allocator_ref allocRef(*this);
		containers::destruct(destr, destrEnd, allocRef.allocator, typename Policy::destruct_tag());
	}
	LEAN_INLINE void open_uninit(Element *where, Element *whereEnd)
	{
		base_type::allocator_ref allocRef(*this);
		containers::open_uninit(where, whereEnd, m_elementsEnd, allocRef.allocator,
			typename Policy::move_tag(), typename Policy::destruct_tag());
	}
	LEAN_INLINE void close_uninit(Element *where, Element *whereEnd)
	{
		base_type::allocator_ref allocRef(*this);
		containers::close_uninit(where, whereEnd, m_elementsEnd, allocRef.allocator,
			typename Policy::move_tag(), typename Policy::destruct_tag());
	}
	LEAN_INLINE void close(Element *where, Element *whereEnd)
	{
		base_type::allocator_ref allocRef(*this);
		containers::close(where, whereEnd, m_elementsEnd, allocRef.allocator,
			typename Policy::move_tag(), typename Policy::destruct_tag());
	}

	template <bool RawMove>
	LEAN_INLINE void reallocate_move_construct_helper(Element *source, Element *sourceEnd, Element *dest)
	{
		move_construct(source, sourceEnd, dest);
	}
	template <>
	LEAN_INLINE void reallocate_move_construct_helper<true>(Element *source, Element *sourceEnd, Element *dest) { }

	/// Allocates space for the given number of elements.
	void reallocate(size_type newCapacity)
	{
		base_type::allocator_ref allocRef(*this);

		Element *newElements = allocRef.allocator.allocate(newCapacity);

		if (!Policy::raw_move)
			try
			{
				reallocate_move_construct_helper<Policy::raw_move>(m_elements, m_elementsEnd, newElements);
			}
			catch(...)
			{
				allocRef.allocator.deallocate(newElements, newCapacity);
				throw;
			}
		else if (!empty())
			// Raw move works by copying bitwise w/o destructing afterwards
			// -> Works for all objects that are not "self-aware" (i.e. most objects)
			memcpy(newElements, m_elements, size() * sizeof(Element));

		Element *oldElements = m_elements;
		Element *oldElementsEnd = m_elementsEnd;
		size_type oldCapacity = capacity();
		
		// ORDER: IMPORTANT: Mind the order, size() based on member variables!
		m_elementsEnd = newElements + size();
		m_capacityEnd = newElements + newCapacity;
		m_elements = newElements;

		if (oldElements)
		{
			// IMPORTANT: Don't destruct on raw move!
			if (!Policy::raw_move)
				// Do nothing on exception, resources leaking anyways!
				destruct(oldElements, oldElementsEnd);
			allocRef.allocator.deallocate(oldElements, oldCapacity);
		}
	}

	/// Frees the given elements.
	LEAN_INLINE void free()
	{
		if (m_elements)
		{
			// Do nothing on exception, resources leaking anyways!
			destruct(m_elements, m_elementsEnd);
			this->allocator().deallocate(m_elements, capacity());
		}
	}

	/// Grows vector storage to fit the given new count.
	LEAN_INLINE void growTo(size_type newCount, bool checkLength = true)
	{
		// Mind overflow
		if (checkLength)
			check_length(newCount);

		reallocate(next_capacity_hint(newCount));
	}
	/// Grows vector storage to fit the given additional number of elements.
	LEAN_INLINE void grow(size_type count)
	{
		size_type oldSize = size();

#ifndef LEAN_OPTIMIZE_NO_OVERFLOW_CHECKS
		// Mind overflow
		if (count > max_size() || max_size() - count < oldSize)
			length_exceeded();
#endif
		LEAN_ASSERT(count <= max_size() && max_size() - count >= oldSize);

		growTo(oldSize + count, false);
	}
	/// Grows vector storage and inserts the given element at the end of the vector.
	LEAN_INLINE Element& grow_and_relocate(Element &value)
	{
		size_type index = lean::addressof(value) - m_elements;
		grow(1);
		
		// Index is unsigned, make use of wrap-around
		return (index < size())
			? m_elements[index]
			: value;
	}

	/// Grows vector storage to fit the given new count, not inlined.
	LEAN_NOINLINE void growToHL(size_type newCount)
	{
		growTo(newCount);
	}
	/// Grows vector storage to fit the given additional number of elements, not inlined.
	LEAN_NOINLINE void growHL(size_type count)
	{
		grow(count);
	}
	/// Grows vector storage and inserts the given element at the end of the vector.
	LEAN_NOINLINE void grow_and_pushHL(const Element &value)
	{
		copy_construct(m_elementsEnd, grow_and_relocate(const_cast<Element&>(value)));
		++m_elementsEnd;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Grows vector storage and inserts the given element at the end of the vector.
	LEAN_NOINLINE void grow_and_pushHL(Element &&value)
	{
		move_construct(m_elementsEnd, grow_and_relocate(value));
		++m_elementsEnd;
	}
#endif

	/// Triggers an out of range error.
	LEAN_NOINLINE static void out_of_range()
	{
		throw std::out_of_range("simple_vector<T> out of range");
	}
	/// Checks the given position.
	LEAN_INLINE void check_pos(size_type pos) const
	{
		if (pos >= size())
			out_of_range();
	}
	/// Triggers a length error.
	LEAN_NOINLINE static void length_exceeded()
	{
		throw std::length_error("simple_vector<T> too long");
	}
	/// Checks the given length.
	LEAN_INLINE void check_length(size_type count)
	{
#ifndef LEAN_OPTIMIZE_NO_OVERFLOW
		if (count > max_size())
			length_exceeded();
#endif
		LEAN_ASSERT(count <= max_size());
	}

public:
	/// Constructs an empty vector.
	simple_vector()
		: m_elements(nullptr),
		m_elementsEnd(nullptr),
		m_capacityEnd(nullptr) { }
	/// Constructs an empty vector.
	explicit simple_vector(allocator_type allocator)
		: base_type(allocator),
		m_elements(nullptr),
		m_elementsEnd(nullptr),
		m_capacityEnd(nullptr) { }
	/// Copies all elements from the given vector to this vector.
	simple_vector(const simple_vector &right)
		: base_type(right),
		m_elements(nullptr),
		m_elementsEnd(nullptr),
		m_capacityEnd(nullptr)
	{
		try
		{
			assign_disjoint(right.begin(), right.end());
		}
		catch (...)
		{
			free();
			throw;
		}
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves all elements from the given vector to this vector.
	simple_vector(simple_vector &&right) noexcept
		: base_type(std::move(right)),
		m_elements(std::move(right.m_elements)),
		m_elementsEnd(std::move(right.m_elementsEnd)),
		m_capacityEnd(std::move(right.m_capacityEnd))
	{
		right.m_elements = nullptr;
		right.m_elementsEnd = nullptr;
		right.m_capacityEnd = nullptr;
	}
#endif
	/// Destroys all elements in this vector.
	~simple_vector()
	{
		free();
	}

	/// Copies all elements of the given vector to this vector.
	simple_vector& operator =(const simple_vector &right)
	{
		if (&right != this)
			assign_disjoint(right.begin(), right.end());
		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves all elements from the given vector to this vector.
	simple_vector& operator =(simple_vector &&right) noexcept
	{
		if (&right != this)
		{
			free();

			m_elements = std::move(right.m_elements);
			m_elementsEnd = std::move(right.m_elementsEnd);
			m_capacityEnd = std::move(right.m_capacityEnd);

			right.m_elements = nullptr;
			right.m_elementsEnd = nullptr;
			right.m_capacityEnd = nullptr;

			this->base_type::operator =(std::move(right));
		}
		return *this;
	}
#endif

	/// Assigns the given range of elements to this vector.
	template <class Iterator>
	void assign(Iterator source, Iterator sourceEnd)
	{
		LEAN_ASSERT(source <= sourceEnd);

		Element *sourceElements = const_cast<Element*>( lean::addressof(*source) );

		// Index is unsigned, make use of wrap-around
		if (static_cast<size_type>(sourceElements - m_elements) < size())
		{
			Element *sourceElementsEnd = const_cast<Element*>( lean::addressof(*sourceEnd) );
			LEAN_ASSERT(sourceElementsEnd <= m_elementsEnd);

			// Move (always back to front)
			LEAN_ASSERT(m_elements <= sourceElements);
			move(sourceElements, sourceElementsEnd, m_elements);

			Element *oldElementsEnd = m_elementsEnd;
			m_elementsEnd = m_elements + (sourceElementsEnd - sourceElements);
			destruct(m_elementsEnd, oldElementsEnd);
		}
		else
			assign_disjoint(source, sourceEnd);
	}
	/// Assigns the given disjoint range of elements to this vector.
	template <class Iterator>
	void assign_disjoint(Iterator source, Iterator sourceEnd)
	{
		// Clear before reallocation to prevent full-range moves
		clear();

		size_type count = sourceEnd - source;

		if (count > capacity())
			growToHL(count);

		copy_construct(source, sourceEnd, m_elements);
		m_elementsEnd = m_elements + count;
	}
	
	/// Returns a pointer to the next non-constructed element.
	LEAN_INLINE void* allocate_back()
	{
		if (m_elementsEnd == m_capacityEnd)
			growHL(1);

		return m_elementsEnd;
	}
	/// Marks the next element as constructed.
	LEAN_INLINE reference shift_back(value_type *newElement)
	{
		LEAN_ASSERT(m_elementsEnd != m_capacityEnd);
		LEAN_ASSERT(newElement == m_elementsEnd);

		return *m_elementsEnd++;
	}

	/// Appends a default-constructed element to this vector.
	LEAN_INLINE reference push_back()
	{
		if (m_elementsEnd == m_capacityEnd)
			growHL(1);

		default_construct(m_elementsEnd);
		return *(m_elementsEnd++);
	}
	/// Appends the given element to this vector.
	LEAN_INLINE void push_back(const value_type &value)
	{
		if (m_elementsEnd == m_capacityEnd)
			grow_and_pushHL(value);
		else
		{
			copy_construct(m_elementsEnd, value);
			++m_elementsEnd;
		}
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Appends the given element to this vector.
	LEAN_INLINE void push_back(value_type &&value)
	{
		if (m_elementsEnd == m_capacityEnd)
			grow_and_pushHL(std::move(value));
		else
		{
			move_construct(m_elementsEnd, value);
			++m_elementsEnd;
		}
	}
#endif
	/// Removes the last element from this vector.
	LEAN_INLINE void pop_back()
	{
		LEAN_ASSERT(!empty());

		destruct(--m_elementsEnd);
	}

	/// Inserts the given element.
	iterator insert(iterator where, const value_type &value)
	{
		LEAN_ASSERT(m_elements <= where);
		LEAN_ASSERT(where <= m_elementsEnd);

		const value_type *safeVal = lean::addressof(value);

		if (m_elementsEnd == m_capacityEnd)
		{
			size_t whereIdx = where - m_elements;
			safeVal = lean::addressof(grow_and_relocate(const_cast<Element&>(value)));
			where = m_elements + whereIdx;
		}

		open_uninit(where, where + 1);

		try
		{
			copy_construct(where, *safeVal);
		}
		catch (...)
		{
			close_uninit(where, where + 1);
			throw;
		}

		return where;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Inserts the given element.
	iterator insert(iterator where, value_type &&value)
	{
		LEAN_ASSERT(m_elements <= where);
		LEAN_ASSERT(where <= m_elementsEnd);

		value_type *safeVal = lean::addressof(value);

		if (m_elementsEnd == m_capacityEnd)
		{
			size_t whereIdx = where - m_elements;
			safeVal = lean::addressof(grow_and_relocate(value));
			where = m_elements + whereIdx;
		}

		open_uninit(where, where + 1);

		try
		{
			move_construct(where, *safeVal);
		}
		catch (...)
		{
			close_uninit(where, where + 1);
			throw;
		}

		return where;
	}
#endif

	/// Inserts the given elements.
	template <class SrcIt>
	iterator insert_disjoint(iterator where, SrcIt begin, SrcIt end)
	{
		LEAN_ASSERT(m_elements <= where);
		LEAN_ASSERT(where <= m_elementsEnd);

		size_t count = end - begin;

		if (static_cast<size_t>(m_capacityEnd - m_elementsEnd) < count)
		{
			size_t whereIdx = where - m_elements;
			grow(count);
			where = m_elements + whereIdx;
		}

		open_uninit(where, where + count);

		try
		{
			copy_construct(begin, end, where);
		}
		catch (...)
		{
			close_uninit(where, where + count);
			throw;
		}

		return where;
	}
	
	/// Erases the given element.
	void erase(iterator where)
	{
		LEAN_ASSERT(m_elements <= where);
		LEAN_ASSERT(where < m_elementsEnd);

		close(where, where + 1);
	}
	/// Erases the given range of elements.
	void erase(iterator where, iterator whereEnd)
	{
		LEAN_ASSERT(m_elements <= where);
		LEAN_ASSERT(whereEnd <= m_elementsEnd);
		LEAN_ASSERT(where <= whereEnd);

		if (where != whereEnd)
			close(where, whereEnd);
	}

	/// Clears all elements from this vector.
	void clear()
	{
		Element *oldElementsEnd = m_elementsEnd;
		m_elementsEnd = m_elements;
		destruct(m_elements, oldElementsEnd);
	}

	/// Reserves space for the predicted number of elements given.
	void reserve(size_type newCapacity)
	{
		// Mind overflow
		check_length(newCapacity);

		if (newCapacity > capacity())
			reallocate(newCapacity);
	}
	/// Reserves space for at least the given predicted number of elements.
	void reserve_grow_to(size_type newCapacity)
	{
		if (newCapacity > capacity())
			growTo(newCapacity);
	}
	/// Reserves space for at least the given predicted number of _additional_ elements.
	void reserve_grow_by(size_type newElements)
	{
		if (newElements > static_cast<size_t>(m_capacityEnd - m_elementsEnd))
			grow(newElements);
	}
	/// Shrinks this vector, removing elements from the back.
	void shrink(size_type newCount)
	{
		if (newCount < size())
		{
			Element *oldElementsEnd = m_elementsEnd;
			m_elementsEnd = m_elements + newCount;
			destruct(m_elementsEnd, oldElementsEnd);
		}
	}
	/// Resizes this vector, either appending empty elements to or removing elements from the back of this vector.
	void resize(size_type newCount)
	{
		if (newCount > size())
		{
			if (newCount > capacity())
				growToHL(newCount);
			
			Element *newElementsEnd = m_elements + newCount;
			default_construct(m_elementsEnd, newElementsEnd);
			m_elementsEnd = newElementsEnd;
		}
		else
			shrink(newCount);
	}
	/// Resizes this vector, either appending empty elements to or removing elements from the back of this vector.
	void resize(size_type newCount, const value_type &value)
	{
		if (newCount > size())
		{
			if (newCount > capacity())
				growToHL(newCount);
			
			Element *newElementsEnd = m_elements + newCount;

			while (m_elementsEnd != newElementsEnd)
			{
				copy_construct(m_elementsEnd, value);
				++m_elementsEnd;
			}
		}
		else
			shrink(newCount);
	}
	
	/// Gets an element by position, access violation on failure.
	LEAN_INLINE reference at(size_type pos) { check_pos(pos); return m_elements[pos]; };
	/// Gets an element by position, access violation on failure.
	LEAN_INLINE const_reference at(size_type pos) const { check_pos(pos); return m_elements[pos]; };
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

	/// Gets a raw data pointer.
	LEAN_INLINE pointer data() { return m_elements; };
	/// Gets a raw data pointer.
	LEAN_INLINE const_pointer data() const { return m_elements; };
	/// Gets a raw data pointer.
	LEAN_INLINE const_pointer cdata() const { return m_elements; };

	/// Returns an iterator to the first element contained by this vector.
	LEAN_INLINE iterator begin(void) { return m_elements; };
	/// Returns a constant iterator to the first element contained by this vector.
	LEAN_INLINE const_iterator begin(void) const { return m_elements; };
	/// Returns an iterator beyond the last element contained by this vector.
	LEAN_INLINE iterator end(void) { return m_elementsEnd; };
	/// Returns a constant iterator beyond the last element contained by this vector.
	LEAN_INLINE const_iterator end(void) const { return m_elementsEnd; };

	/// Gets a copy of the allocator used by this vector.
	LEAN_INLINE allocator_type get_allocator() const { return this->allocator(); };

	/// Returns true if the vector is empty.
	LEAN_INLINE bool empty(void) const { return (m_elements == m_elementsEnd); };
	/// Returns the number of elements contained by this vector.
	LEAN_INLINE size_type size(void) const { return m_elementsEnd - m_elements; };
	/// Returns the number of elements this vector could contain without reallocation.
	LEAN_INLINE size_type capacity(void) const { return m_capacityEnd - m_elements; };

	/// Computes a new capacity based on the given number of elements to be stored.
	size_type next_capacity_hint(size_type count) const
	{
		size_type capacity = this->capacity();
		size_type maxSize = this->max_size();

		LEAN_ASSERT(capacity <= maxSize);
		LEAN_ASSERT(count <= maxSize);
		
		size_type capacityDelta = capacity / 2;

		// Try to increase capacity by 1.5 (mind overflow)
		capacity = (maxSize - capacityDelta < capacity)
			? maxSize
			: capacity + capacityDelta;

		// Fit to count, if greater than next capacity step
		if (capacity < count)
			capacity = count;

		// Always allocate minimum number of 16 elements?
//		if (capacity < 16)
//			capacity = 16;
		
		return capacity;
	}

	/// Estimates the maximum number of elements that may be constructed.
	LEAN_INLINE size_type max_size() const
	{
		return static_cast<size_type>(-1) / sizeof(Element);
	}

	/// Swaps the contents of this vector and the given vector.
	LEAN_INLINE void swap(simple_vector &right) noexcept
	{
		using std::swap;

		this->base_type::swap(right);
		swap(m_elements, right.m_elements);
		swap(m_elementsEnd, right.m_elementsEnd);
		swap(m_capacityEnd, right.m_capacityEnd);
	}
};

/// Swaps the contents of the given vectors.
template <class Element, class Policy, class Allocator>
LEAN_INLINE void swap(simple_vector<Element, Policy, Allocator> &left, simple_vector<Element, Policy, Allocator> &right) noexcept
{
	left.swap(right);
}

/// Default vector binder.
template <class Policy = vector_policies::nonpod>
struct simple_vector_binder
{
	/// Constructs a vector type from the given element type.
	template <class Type>
	struct rebind
	{
		typedef heap_allocator<Type> allocator_type;
		typedef Policy policy;
		typedef simple_vector<Type, policy, allocator_type> type;
	};
};

} // namespace

namespace simple_vector_policies = containers::simple_vector_policies;
using containers::simple_vector;

using containers::simple_vector_binder;

} // namespace

#endif