/*****************************************************/
/* lean Tags                    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_TAGS_FORWARD_REF
#define LEAN_TAGS_FORWARD_REF

#include "../lean.h"

namespace lean
{
namespace tags
{

/// Reference wrapper class that allows the storage of both r- and const l-value references.
template <class Type>
class forward_ref
{
	template <class T2>
	friend class forward_ref;

public:
	typedef Type value_type;

private:
	const value_type *p;
	bool is_rval;

public:
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// R-value constructor.
	LEAN_INLINE forward_ref(value_type &&v)
		: p(lean::addressof(v)), is_rval(true) { }
#endif
	/// L-value constructor.
	LEAN_INLINE forward_ref(const value_type &v)
		: p(lean::addressof(v)), is_rval(false) { }
	/// Implicitly casting copy constructor.
	template <class T2>
	LEAN_INLINE forward_ref(const forward_ref<T2> &right)
		: p(right.p), is_rval(right.is_rval) { }

#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Gets whether this wrapper stores an r-value reference.
	LEAN_INLINE bool rval() const { return is_rval; }
	/// Gets an r-value reference to the stored object.
	LEAN_INLINE value_type&& move() { return std::move( *const_cast<value_type*>(p) ); }
#else
	LEAN_INLINE bool rval() const { return false; }
	LEAN_INLINE const value_type& move() { return *p; }
#endif
	/// Gets a const l-value reference to the stored object.
	LEAN_INLINE const value_type& copy() const { return *p; }
};

} // namespace

using tags::forward_ref;

} // namespace

#endif