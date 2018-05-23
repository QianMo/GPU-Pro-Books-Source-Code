/*****************************************************/
/* lean Containers              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONTAINERS_STATIC_ARRAY
#define LEAN_CONTAINERS_STATIC_ARRAY

#include "../lean.h"
#include "../functional/variadic.h"
#include "../meta/type_traits.h"
#include "construction.h"

namespace lean 
{
namespace containers
{

/// Static array class.
template <class Element, size_t Capacity>
class static_array
{
public:
	/// Type of the size returned by this vector.
	typedef size_t size_type;
	/// Size.
	static const size_type capacity = Capacity;

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
	// Make sure size_type is unsigned
	LEAN_STATIC_ASSERT(is_unsigned<size_type>::value);
		
	char m_memory[sizeof(value_type) * capacity];
	value_type *m_elementsEnd;
	
	/// First element in the array.
	value_type *const elements() { return reinterpret_cast<value_type*>(&m_memory[0]); }
	/// First element in the array.
	const value_type *const elements() const { return reinterpret_cast<const value_type*>(&m_memory[0]); }
	/// One past the last element in the array.
	value_type*& elementsEnd() { return m_elementsEnd; }
	/// One past the last element in the array.
	const value_type *const& elementsEnd() const { return m_elementsEnd; }
	
	/// Moves elements from the given source range to the given destination.
	template <class Iterator>
	static void swap(value_type *left, value_type *leftEnd, value_type *right)
	{
		using std::swap;

		for (; left != leftEnd; ++left, ++right)
			swap(*left, *right);
	}

public:
	/// Constructs an empty vector.
	static_array()
		: m_elementsEnd(elements()) { }
	/// Copies all elements from the given vector to this vector.
	static_array(const static_array &right)
		: m_elementsEnd( containers::copy_construct(right.elements(), right.elementsEnd(), elements(), no_allocator) ) { }
	/// Copies all elements from the given vector to this vector.
	template <size_t RightCapacity>
	explicit static_array(const static_array<value_type, RightCapacity> &right)
	{
		LEAN_ASSERT(right.size() <= capacity);
		elementsEnd() = containers::copy_construct(right.elements(), right.elementsEnd(), elements(), no_allocator);
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves all elements from the given vector to this vector.
	static_array(static_array &&right)
		: m_elementsEnd( containers::move_construct(right.elements(), right.elementsEnd(), elements(), no_allocator) ) { }
	/// Moves all elements from the given vector to this vector.
	template <size_t RightCapacity>
	explicit static_array(static_array<value_type, RightCapacity> &&right)
	{
		LEAN_ASSERT(right.size() <= capacity);
		elementsEnd() = containers::move_construct(right.elements(), right.elementsEnd(), elements(), no_allocator);
	}
#endif
	/// Moves all elements from the given vector to this vector.
	template <size_t RightCapacity>
	static_array(static_array<value_type, RightCapacity> &right, consume_t)
	{
		LEAN_ASSERT(right.size() <= capacity);
		elementsEnd() = containers::move_construct(right.elements(), right.elementsEnd(), elements(), no_allocator);
	}
	/// Copies all elements from the given range to this vector.
	template <class Iterator>
	static_array(Iterator begin, Iterator end)
		: m_elementsEnd( containers::copy_construct(begin, end, elements(), no_allocator) ) { }
	/// Moves all elements from the given range to this vector.
	template <class Iterator>
	static_array(Iterator begin, Iterator end, consume_t)
		: m_elementsEnd( containers::move_construct(begin, end, elements(), no_allocator) ) {  }
	/// Destroys all elements in this vector.
	~static_array()
	{
		containers::destruct(elements(), elementsEnd(), no_allocator);
	}

	/// Copies all elements of the given vector to this vector.
	template <size_t RightCapacity>
	void assign(const static_array<value_type, RightCapacity> &right)
	{
		if (&right != this)
		{
			clear();
			LEAN_ASSERT(right.size() <= capacity);
			elementsEnd() = containers::copy_construct(right.elements(), right.elementsEnd(), elements(), no_allocator);
		}
	}
	/// Moves all elements from the given vector to this vector.
	template <size_t RightCapacity>
	void assign(static_array<value_type, RightCapacity> &right, consume_t)
	{
		if (&right != this)
		{
			clear();
			LEAN_ASSERT(right.size() <= capacity);
			elementsEnd() = containers::move_construct(right.elements(), right.elementsEnd(), elements(), no_allocator);
		}
	}

	/// Copies all elements of the given vector to this vector.
	LEAN_INLINE static_array& operator =(const static_array &right)
	{
		assign(right);
		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves all elements from the given vector to this vector.
	LEAN_INLINE static_array& operator =(static_array &&right)
	{
		assign(right, consume);
		return *this;
	}
	/// Moves all elements from the given vector to this vector.
	template <size_t RightCapacity>
	LEAN_INLINE void assign(static_array<value_type, RightCapacity> &&right)
	{
		assign(right, consume);
	}
#endif
	
	/// Copies all elements from the given range to this vector.
	template <class Iterator>
	void assign_disjoint(Iterator begin, Iterator end)
	{
		clear();
		elementsEnd() = containers::copy_construct(begin, end, elements(), no_allocator);
	}
	/// Moves all elements from the given range to this vector.
	template <class Iterator>
	void assign_disjoint(Iterator begin, Iterator end, consume_t)
	{
		clear();
		elementsEnd() = containers::move_construct(begin, end, elements(), no_allocator);
	}

	/// Returns a pointer to the next non-constructed element.
	LEAN_INLINE void* allocate_back()
	{
		return elementsEnd();
	}
	/// Marks the next element as constructed.
	LEAN_INLINE reference shift_back(value_type *newElement)
	{
		LEAN_ASSERT(newElement == elementsEnd());

		return *elementsEnd()++;
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
		containers::default_construct(elementsEnd(), no_allocator);
		return *elementsEnd()++;
	}
	/// Appends a default-constructed element to this vector.
	LEAN_INLINE pointer push_back_n(size_type count)
	{
		Element *firstElement = elementsEnd();
		containers::default_construct(firstElement, firstElement + count, no_allocator);
		elementsEnd() += count;
		return firstElement;
	}
	/// Appends the given element to this vector.
	LEAN_INLINE reference push_back(const value_type &value)
	{
		containers::copy_construct(elementsEnd(), value, no_allocator);
		return *elementsEnd()++;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Appends the given element to this vector.
	LEAN_INLINE reference push_back(value_type &&value)
	{
		containers::move_construct(elementsEnd(), value, no_allocator);
		return *elementsEnd()++;
	}
#endif
	/// Removes the last element from this vector.
	LEAN_INLINE void pop_back()
	{
		LEAN_ASSERT(!empty());

		containers::destruct(--elementsEnd(), no_allocator);
	}

	/// Clears all elements from this vector.
	LEAN_INLINE void clear()
	{
		Element *oldElementsEnd = elementsEnd();
		elementsEnd() = elements();
		containers::destruct(elements(), oldElementsEnd, no_allocator);
	}
	
	/// Gets the first element in the vector, access violation on failure.
	LEAN_INLINE reference front(void) { LEAN_ASSERT(!empty()); return *elements(); };
	/// Gets the first element in the vector, access violation on failure.
	LEAN_INLINE const_reference front(void) const { LEAN_ASSERT(!empty()); return *elements(); };
	/// Gets the last element in the vector, access violation on failure.
	LEAN_INLINE reference back(void) { LEAN_ASSERT(!empty()); return elementsEnd()[-1]; };
	/// Gets the last element in the vector, access violation on failure.
	LEAN_INLINE const_reference back(void) const { LEAN_ASSERT(!empty()); return elementsEnd()[-1]; };

	/// Gets an element by position, access violation on failure.
	LEAN_INLINE reference operator [](size_type pos) { return elements()[pos]; };
	/// Gets an element by position, access violation on failure.
	LEAN_INLINE const_reference operator [](size_type pos) const { return elements()[pos]; };

	/// Returns an iterator to the first element contained by this vector.
	LEAN_INLINE iterator begin(void) { return elements(); };
	/// Returns a constant iterator to the first element contained by this vector.
	LEAN_INLINE const_iterator begin(void) const { return elements(); };
	/// Returns an iterator beyond the last element contained by this vector.
	LEAN_INLINE iterator end(void) { return elementsEnd(); };
	/// Returns a constant iterator beyond the last element contained by this vector.
	LEAN_INLINE const_iterator end(void) const { return elementsEnd(); };

	/// Returns true if the vector is empty.
	LEAN_INLINE bool empty(void) const { return (elements() == elementsEnd()); };
	/// Returns the number of elements contained by this vector.
	LEAN_INLINE size_type size(void) const { return elementsEnd() - elements(); };

	/// Swaps the contents of this vector and the given vector.
	template <size_t RightCapacity>
	LEAN_INLINE void swap(static_array<value_type, RightCapacity> &right)
	{
		value_type *min, *minEnd, *max, *maxEnd;

		if (size() < right.size())
		{
			min = elements();
			minEnd = elementsEnd();
			max = right.elements();
			maxEnd = right.elementsEnd();
		}
		else
		{
			min = right.elements();
			minEnd = right.elementsEnd();
			max = elements();
			maxEnd = elementsEnd();
		}

		swap(min, minEnd, max);
		containers::move_construct(max + (minEnd - min), maxEnd, minEnd, no_allocator);
		
		size_type leftSize = right.size();
		size_type rightSize = size();

		elementsEnd() = elements() + leftSize;
		right.elementsEnd() = right.elements() + rightSize;

		containers::destruct(max + (minEnd - min), maxEnd, no_allocator);
	}
};

/// Swaps the contents of the given arrays.
template <class Element, size_t LeftCapacity, size_t RightCapacity>
LEAN_INLINE void swap(static_array<Element, LeftCapacity> &left, static_array<Element, RightCapacity> &right)
{
	left.swap(right);
}

} // namespace

using containers::static_array;

} // namespace

#endif