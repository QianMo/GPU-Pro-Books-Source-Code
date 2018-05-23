/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_SCOPE_GUARD
#define LEAN_SMART_SCOPE_GUARD

#include "../lean.h"
#include "../tags/noncopyable.h"
#include "../functional/callable.h"

namespace lean
{
namespace smart
{

/// Stores and calls a callable object on destruction.
class scope_annex_base : public nonassignable
{
private:
	mutable bool m_valid;

protected:
	/// Constructs a valid scope annex.
	LEAN_INLINE scope_annex_base()
		: m_valid(true) { }
	/// Copies AND INVALIDATES the given scope annex.
	LEAN_INLINE scope_annex_base(const scope_annex_base &right)
		: m_valid(true)
	{
		right.m_valid = false;
	}
	LEAN_INLINE ~scope_annex_base() { }

	/// Gets whether this instance is still valid.
	LEAN_INLINE bool valid() const { return m_valid; }
};

/// Stores and calls a callable object on destruction.
template <class Callable>
class scope_annex_impl : public scope_annex_base
{
private:
	Callable m_callable;

public:
	/// Stores the given callable, to be called on destruction.
	LEAN_INLINE explicit scope_annex_impl(const Callable &callable)
		: m_callable(callable) { }
	/// Calls the callable stored by this guard.
	LEAN_INLINE ~scope_annex_impl()
	{
		if (valid())
			m_callable();
	}
};

/// Convenience type for temporary scope annex variables.
typedef const scope_annex_base& scope_annex;

/// Constructs a scope annex object from the given callable object.
template <class Callable>
LEAN_INLINE scope_annex_impl<Callable> make_scope_annex(const Callable& callable)
{
	return scope_annex_impl<Callable>(callable);
}

/// Constructs a scope annex object from the given function pointer.
template <class Signature>
LEAN_INLINE scope_annex_impl< callable_fun<Signature> > make_scope_annex(Signature *fun)
{
	return make_scope_annex( make_callable(fun) );
}

/// Constructs a scope annex object from the given object and method pointer.
template <class Class, class Signature>
LEAN_INLINE scope_annex_impl< callable_memfun<Class, Signature> > make_scope_annex(Class *obj, Signature Class::*fun)
{
	return make_scope_annex( make_callable(obj, fun) );
}

/// Stores and calls a callable object on destruction, if not disarmed.
class scope_guard_base : public nonassignable
{
private:
	mutable bool m_armed;

protected:
	/// Constructs a scope guard (optionally disarmed).
	LEAN_INLINE explicit scope_guard_base(bool arm = true)
		: m_armed(arm) { }
	/// Copies AND DISARMS the given scope guard.
	LEAN_INLINE scope_guard_base(const scope_guard_base &right)
		: m_armed(right.m_armed)
	{
		right.disarm();
	}
	LEAN_INLINE ~scope_guard_base() { }

public:
	/// Sets whether the scope guard is currently armed.
	LEAN_INLINE void armed(bool arm) const { m_armed = arm; }
	/// Gets whether the scope guard is currently armed.
	LEAN_INLINE bool armed() const { return m_armed; }

	/// Disarms this scope guard.
	LEAN_INLINE void disarm() const { armed(false); }
	/// Re-arms this scope guard.
	LEAN_INLINE void arm() const { armed(true); }
};

/// Convenience type for temporary scope guard variables.
typedef const scope_guard_base& scope_guard;

/// Stores and calls a callable object on destruction, if not disarmed.
template <class Callable>
class scope_guard_impl : public scope_guard_base
{
private:
	Callable m_callable;

public:
	/// Stores the given callable, to be called on destruction, if not disarmed.
	LEAN_INLINE explicit scope_guard_impl(const Callable &callable, bool arm = true)
		: scope_guard_base(arm),
		m_callable(callable) { }
	/// Calls the callable stored by this guard, if not disarmed.
	LEAN_INLINE ~scope_guard_impl()
	{
		if (armed())
			m_callable();
	}
};

/// Constructs a scope guard from the given callable object.
template <class Callable>
LEAN_INLINE scope_guard_impl<Callable> make_scope_guard(const Callable& callable)
{
	return scope_guard_impl<Callable>(callable);
}

/// Constructs a scope guard from the given function pointer.
template <class Signature>
LEAN_INLINE scope_guard_impl< callable_fun<Signature> > make_scope_guard(Signature *fun)
{
	return make_scope_guard( make_callable(fun) );
}

/// Constructs a scope guard from the given object and method pointer.
template <class Class, class Signature>
LEAN_INLINE scope_guard_impl< callable_memfun<Class, Signature> > make_scope_guard(Class *obj, Signature Class::*fun)
{
	return make_scope_guard( make_callable(obj, fun) );
}

} // namespace

using smart::scope_annex;
using smart::make_scope_annex;

using smart::scope_guard;
using smart::make_scope_guard;

} // namespace

#endif