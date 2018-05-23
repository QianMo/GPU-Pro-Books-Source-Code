/*****************************************************/
/* lean Containers              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONTAINERS_SIMPLE_QUEUE
#define LEAN_CONTAINERS_SIMPLE_QUEUE

#include "../lean.h"
#include "../concurrent/critical_section.h"

namespace lean 
{
namespace containers
{

/// Simple thread-safe queue class
template < class Container >
class simple_queue
{
public:
	/// Type of the container wrapped.
	typedef Container container_type;
	/// Type of the size returned by this queue.
	typedef typename container_type::size_type size_type;

	/// Type of pointers to the elements contained by this queue.
	typedef typename container_type::pointer pointer;
	/// Type of constant pointers to the elements contained by this queue.
	typedef typename container_type::const_pointer const_pointer;
	/// Type of references to the elements contained by this queue.
	typedef typename container_type::reference reference;
	/// Type of constant references to the elements contained by this queue.
	typedef typename container_type::const_reference const_reference;
	/// Type of the elements contained by this queue.
	typedef typename container_type::value_type value_type;

private:
	mutable critical_section m_lock;
	container_type m_elements;

public:
	/// Constructs an empty queue.
	simple_queue() { }
	/// Copies all elements from the given vector to this queue.
	simple_queue(const container_type &right)
		: m_elements(right) { }
	/// Copies all elements from the given queue to this queue.
	simple_queue(const simple_queue &right)
		: m_elements(right.m_elements) { }

	/// Copies all elements of the given queue to this queue.
	simple_queue& operator =(const simple_queue &right)
	{
		scoped_cs_lock lock(m_lock);
		m_elements = right.m_elements;
		return *this;
	}

	/// Gets the wrapped vector.
	LEAN_INLINE container_type& container() { return m_elements; }
	/// Gets the wrapped vector.
	LEAN_INLINE const container_type& container() const { return m_elements; }

	/// Gets the lock.
	LEAN_INLINE critical_section& lock() { return m_lock; }

	/// Appends the given element to this queue.
	void push_back(const value_type &value)
	{
		scoped_cs_lock lock(m_lock);
		m_elements.push_back(value);
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Appends the given element to this queue.
	void push_back(value_type &&value)
	{
		scoped_cs_lock lock(m_lock);
		m_elements.push_back( LEAN_MOVE(value) );
	}
#endif
	/// Removes the last element from this queue.
	value_type pop_back()
	{
		scoped_cs_lock lock(m_lock);
		LEAN_ASSERT(!empty());

		value_type value = LEAN_MOVE(m_elements.back());
		m_elements.pop_back();
		return LEAN_MOVE(value);
	}
	/// Removes the first element from this queue.
	value_type pop_front()
	{
		scoped_cs_lock lock(m_lock);
		LEAN_ASSERT(!empty());

		value_type value = LEAN_MOVE(m_elements.front());
		m_elements.pop_front();
		return LEAN_MOVE(value);
	}

	/// Clears all elements from this queue.
	void clear()
	{
		scoped_cs_lock lock(m_lock);
		m_elements.clear();
	}

	/// Reserves space for the predicted number of elements given.
	void reserve(size_type newCapacity)
	{
		scoped_cs_lock lock(m_lock);
		m_elements.reserve(newCapacity);
	}

	/// Returns true if the queue is empty.
	bool empty(void) const { scoped_cs_lock lock(m_lock); return m_elements.empty(); };
	/// Returns the number of elements contained by this queue.
	size_type size(void) const { scoped_cs_lock lock(m_lock); return m_elements.size(); };
	/// Returns the number of elements this queue could contain without reallocation.
	size_type capacity(void) const { scoped_cs_lock lock(m_lock); return m_elements.capacity(); };

	/// Estimates the maximum number of elements that may be constructed.
	LEAN_INLINE size_type max_size() const { return m_elements.max_size(); }
};

} // namespace

using containers::simple_queue;

} // namespace

#endif