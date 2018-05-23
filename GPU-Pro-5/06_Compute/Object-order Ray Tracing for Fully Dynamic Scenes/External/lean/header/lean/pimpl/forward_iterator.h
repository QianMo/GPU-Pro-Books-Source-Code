/*****************************************************/
/* lean PImpl                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PIMPL_FORWARD_ITERATOR
#define LEAN_PIMPL_FORWARD_ITERATOR

#include "../lean.h"
#include "../meta/strip.h"

namespace lean
{
namespace pimpl
{

/// Opaque value class that stores iterators of the given prototype & forward-declared container type.
template <class Prototype, class Container, class Element>
class forward_iterator
{
private:
	char m_memory[sizeof(Prototype)];

public:
	/// Type of the value returned when an iterator is dereferenced.
	typedef typename Element deref_type;
	/// Type of the values stored by the given container.
	typedef typename lean::strip_const<Element>::type value_type;
	/// Type of the container.
	typedef typename lean::strip_const<Container>::type container_type;

	/// Actual iterator wrapper.
	struct iterator_wrapper
	{
		/// Type of the iterators wrapped.
		typedef typename lean::conditional_type<
			lean::strip_const<deref_type>::stripped,
			typename container_type::const_iterator,
			typename container_type::iterator >::type iterator;

		LEAN_STATIC_ASSERT_MSG_ALT( sizeof(Prototype) == sizeof(iterator),
			"Iterator prototype required to be of the same size as the actual iterator type.",
			Iterator_prototype_required_to_be_of_the_same_size_as_the_actual_iterator_type );

		/// Wrapped iterator.
		iterator &it;

		/// Wraps the given iterator.
		LEAN_INLINE iterator_wrapper(iterator &it) : it(it) { }
		/// Wraps the given iterator.
		LEAN_INLINE iterator_wrapper(iterator_wrapper &right) : it(right.it) { }

		/// Unwraps the stored iterator.
		LEAN_INLINE operator iterator&() { return it; }
		/// Unwraps the stored iterator.
		LEAN_INLINE operator const iterator&() const { return it; }

	private:
		friend class forward_iterator;

		LEAN_INLINE static iterator& as_iterator(char *mem) { return *reinterpret_cast<iterator*>(mem); }
		LEAN_INLINE static const iterator& as_iterator(const char *mem) { return *reinterpret_cast<const iterator*>(mem); }
		
		template <class Iterator>
		LEAN_INLINE static void destruct(Iterator &it) { it.~Iterator(); }
	};

	/// Iterator default construction.
	LEAN_INLINE forward_iterator()
	{
		new (m_memory) typename iterator_wrapper::iterator();
	}
	/// Copies the given iterator.
	template <class Iterator>
	LEAN_INLINE forward_iterator(Iterator right)
	{
		new (m_memory) typename iterator_wrapper::iterator( LEAN_MOVE(right) );
	}
	/// Copies the given iterator.
	LEAN_INLINE forward_iterator(const forward_iterator &right)
	{
		new (m_memory) typename iterator_wrapper::iterator( iterator_wrapper::as_iterator(right.m_memory) );
	}
	/// Destructs the wrapped iterator.
	LEAN_INLINE ~forward_iterator()
	{
		iterator_wrapper::destruct( iterator_wrapper::as_iterator(m_memory) );
	}
	
	/// Assigns the given iterator.
	template <class Iterator>
	LEAN_INLINE forward_iterator& operator =(Iterator right)
	{
		iterator_wrapper::as_iterator(m_memory) = LEAN_MOVE(right);
		return *this;
	}
	/// Assigns the given iterator.
	LEAN_INLINE forward_iterator& operator =(const forward_iterator &right)
	{
		iterator_wrapper::as_iterator(m_memory) = iterator_wrapper::as_iterator(right.m_memory);
		return *this;
	}

	/// Increments the wrapped iterator.
	LEAN_INLINE forward_iterator& operator ++() { ++iterator_wrapper::as_iterator(m_memory); return *this; }
	/// Decrements the wrapped iterator.
	LEAN_INLINE forward_iterator& operator --() { --iterator_wrapper::as_iterator(m_memory); return *this; }
	/// Increments the wrapped iterator.
	LEAN_INLINE forward_iterator operator ++(int) { forward_iterator prev(*this); ++(*this); return prev; }
	/// Decrements the wrapped iterator.
	LEAN_INLINE forward_iterator operator --(int) { forward_iterator prev(*this); --(*this); return prev; }

	/// Gets a reference wrapper to the wrapped iterator.
	LEAN_INLINE iterator_wrapper get()
	{
		return iterator_wrapper( iterator_wrapper::as_iterator(m_memory) );
	}
	/// Gets a reference wrapper to the wrapped iterator.
	LEAN_INLINE const iterator_wrapper get() const
	{
		return iterator_wrapper( const_cast<typename iterator_wrapper::iterator&>( iterator_wrapper::as_iterator(m_memory) ) );
	}

	/// Dereferences the wrapped iterator.
	LEAN_INLINE Element& operator *() const { return *iterator_wrapper::as_iterator(m_memory); }
	/// Dereferences the wrapped iterator.
	LEAN_INLINE Element* operator ->() const { return iterator_wrapper::as_iterator(m_memory).operator ->(); }

	/// Compares the wrapped iterators.
	LEAN_INLINE bool operator <(const forward_iterator &right) const
	{
		return (iterator_wrapper::as_iterator(m_memory) < iterator_wrapper::as_iterator(right.m_memory));
	}
	/// Compares the wrapped iterators.
	LEAN_INLINE bool operator ==(const forward_iterator &right) const
	{
		return (iterator_wrapper::as_iterator(m_memory) == iterator_wrapper::as_iterator(right.m_memory));
	}
	/// Compares the wrapped iterators.
	LEAN_INLINE bool operator !=(const forward_iterator &right) const
	{
		return !(*this == right);
	}
};

} // namespace

using pimpl::forward_iterator;

} // namespace

#endif