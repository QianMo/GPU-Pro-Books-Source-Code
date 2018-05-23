/*****************************************************/
/* lean Containers              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_TAGS_STRIDED_PTR
#define LEAN_TAGS_STRIDED_PTR

#include "../lean.h"

namespace std { struct random_access_iterator_tag; }

namespace lean
{
namespace tags
{

/// Transitive pointer class that applies pointer const modifiers to the objects pointed to.
template <class Type>
class strided_ptr
{
public:
	/// Type of the elements pointed to.
	typedef Type value_type;
	/// Pointer to the elements.
	typedef value_type* pointer;
	/// Reference to an element.
	typedef value_type& reference;
	/// Difference type.
	typedef ptrdiff_t difference_type;
	/// Random access iterator.
	typedef std::random_access_iterator_tag iterator_category;

private:
	pointer m_ptr;
	difference_type m_stride;

public:
	/// Constructs a strided pointer from the given pointer.
	LEAN_INLINE strided_ptr()
		: m_ptr( nullptr ),
		m_stride( sizeof(value_type) ) { }
	/// Constructs a strided pointer from the given pointer.
	LEAN_INLINE strided_ptr(pointer object, difference_type stride)
		: m_ptr( object ),
		m_stride( stride ) { }
	/// Constructs a strided pointer from the given pointer.
	template <class Type2>
	LEAN_INLINE strided_ptr(Type2 *object, difference_type stride = sizeof(Type2))
		: m_ptr( object ),
		m_stride( stride ) { }
	/// Constructs a strided pointer from the given pointer.
	template <class Type2>
	LEAN_INLINE strided_ptr(const strided_ptr<Type2> &right)
		: m_ptr( right.get() ),
		m_stride( right.get_stride() ) { }
	
	/// Replaces the stored pointer with the given pointer.
	template <class Type2>
	LEAN_INLINE strided_ptr& operator =(const strided_ptr<Type2> &right)
	{
		m_ptr = right.get();
		m_stride = right.get_stride();
		return *this;
	}

	/// Increments the pointer.
	LEAN_INLINE strided_ptr& operator ++() { m_ptr = lean::addressof((*this)[1]); return *this; }
	/// Decrements the pointer.
	LEAN_INLINE strided_ptr& operator --() { m_ptr = lean::addressof((*this)[-1]); return *this; }

	/// Increments the pointer.
	LEAN_INLINE strided_ptr operator ++(int) { strided_ptr old(*this); m_ptr = lean::addressof((*this)[1]); return old; }
	/// Decrements the pointer.
	LEAN_INLINE strided_ptr operator --(int) { strided_ptr old(*this); m_ptr = lean::addressof((*this)[-1]); return old; }

	/// Gets the pointer stored by this strided pointer. Don't call unless you know what you are doing!
	LEAN_INLINE pointer get() const { return m_ptr; }
	/// Gets the stride stored by this strided pointer. Don't call unless you know what you are doing!
	LEAN_INLINE difference_type get_stride() const { return m_stride; }

	/// Gets the first object stored by this strided pointer.
	LEAN_INLINE reference operator *() const { return *m_ptr; }
	/// Gets the first object stored by this strided pointer.
	LEAN_INLINE pointer operator ->() const { return m_ptr; }

	/// Gets the n-th object stored by this strided pointer.
	LEAN_INLINE reference operator [](difference_type n) const
	{
		return *(pointer) ((char*) m_ptr + m_stride * n);
	}
};

template <class T> LEAN_INLINE strided_ptr<T> operator +(const strided_ptr<T> &p, ptrdiff_t diff) { return strided_ptr<T>( lean::addressof(p[diff]), p.get_stride() ); }
template <class T> LEAN_INLINE strided_ptr<T> operator -(const strided_ptr<T> &p, ptrdiff_t diff) { return strided_ptr<T>( lean::addressof(p[-diff]), p.get_stride() ); }

template <class T> LEAN_INLINE bool operator ==(const strided_ptr<T> &l, const strided_ptr<T> &r) { return l.get() == r.get(); }
template <class T> LEAN_INLINE bool operator !=(const strided_ptr<T> &l, const strided_ptr<T> &r) { return l.get() != r.get(); }
template <class T> LEAN_INLINE bool operator <=(const strided_ptr<T> &l, const strided_ptr<T> &r) { return l.get() <= r.get(); }
template <class T> LEAN_INLINE bool operator >=(const strided_ptr<T> &l, const strided_ptr<T> &r) { return l.get() >= r.get(); }
template <class T> LEAN_INLINE bool operator <(const strided_ptr<T> &l, const strided_ptr<T> &r) { return l.get() < r.get(); }
template <class T> LEAN_INLINE bool operator >(const strided_ptr<T> &l, const strided_ptr<T> &r) { return l.get() > r.get(); }

template <class T>
LEAN_INLINE ptrdiff_t operator -(const strided_ptr<T> &p, const strided_ptr<T> &q)
{
	LEAN_ASSERT(p.get_stride() == q.get_stride());
	return ( (char*) p.get() - (char*) q.get() ) / p.get_stride();
}

} // namespace

using tags::strided_ptr;

} // namespace

#endif