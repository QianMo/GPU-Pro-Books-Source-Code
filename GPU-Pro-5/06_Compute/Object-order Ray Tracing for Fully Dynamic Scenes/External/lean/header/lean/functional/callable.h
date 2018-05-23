/*****************************************************/
/* lean Functional              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_FUNCTIONAL_CALLABLE
#define LEAN_FUNCTIONAL_CALLABLE

#include "../lean.h"
#include "../tags/noncopyable.h"
#include "../functional/variadic.h"

namespace lean
{
namespace functional
{

/// Stores the pointer to a function to be called on invokation of operator ().
template <class Signature>
class callable_fun
{
private:
	Signature *m_fun;

public:
	/// Stores the given function to be called by operator().
	LEAN_INLINE callable_fun(Signature *fun)
		: m_fun(fun)
	{
		LEAN_ASSERT(m_fun);
	}
	/// Calls the function stored by this callable object.
	LEAN_INLINE void operator ()()
	{
		(*m_fun)();
	}
};

/// Stores an object and a pointer to a method to be called on invokation of operator ().
template <class Class, class Signature>
class callable_objfun
{
private:
	Class *m_obj;
	Signature *m_fun;

public:
	/// Stores the given object and method to be called by operator().
	LEAN_INLINE callable_objfun(Class *obj, Signature *fun)
		: m_obj(obj),
		m_fun(fun)
	{
		LEAN_ASSERT(m_obj);
		LEAN_ASSERT(m_fun);
	}
	/// Calls the function stored by this callable object.
	LEAN_INLINE void operator ()()
	{
		(*m_fun)(*m_obj);
	}
};

/// Stores an object and a pointer to a method to be called on invokation of operator ().
template <class Class, class Signature>
class callable_memfun
{
private:
	Class *m_obj;
	Signature Class::*m_fun;

public:
	/// Stores the given object and method to be called by operator().
	LEAN_INLINE callable_memfun(Class *obj, Signature Class::*fun)
		: m_obj(obj),
		m_fun(fun)
	{
		LEAN_ASSERT(m_obj);
		LEAN_ASSERT(m_fun);
	}
	/// Calls the function stored by this callable object.
	LEAN_INLINE void operator ()()
	{
		(m_obj->*m_fun)();
	}
};

/// Constructs a callable object from the given function pointer.
template <class Signature>
LEAN_INLINE callable_fun<Signature> make_callable(Signature *fun)
{
	return callable_fun<Signature>(fun);
}

/// Constructs a callable object from the given object and method pointer.
template <class Class, class Signature>
LEAN_INLINE callable_objfun<Class, Signature> make_callable(Class *obj, Signature *fun)
{
	return callable_objfun<Class, Signature>(obj, fun);
}

/// Constructs a callable object from the given object and method pointer.
template <class Class, class Signature>
LEAN_INLINE callable_memfun<Class, Signature> make_callable(Class *obj, Signature Class::*fun)
{
	return callable_memfun<Class, Signature>(obj, fun);
}

/// Stores a pointer to a polymorphic method to be called on invokation of operator ().
template <class Signature>
class vcallable
{
public:
	/// Signature.
	typedef Signature signature_type;
	/// Polymorphic function pointer type.
	typedef signature_type (vcallable::*fun_ptr);
	
private:
	fun_ptr m_fun;

protected:
	/// Stores the given polymorphic method to be called by operator().
	LEAN_INLINE vcallable(fun_ptr fun) noexcept
		: m_fun(fun)
	{
		LEAN_ASSERT(m_fun);
	}
	/// Stores the given polymorphic method to be called by operator().
	template <class Derived>
	LEAN_INLINE vcallable(signature_type Derived::* fun) noexcept
		: m_fun( static_cast<signature_type vcallable::*>(fun) )
	{
		LEAN_ASSERT(m_fun);

		// MONITOR: Does static cast account for that?
		// Make sure cast is "valid"
		LEAN_STATIC_ASSERT_MSG_ALT(
				offsetof(Derived, m_fun) == offsetof(vcallable, m_fun),
				"Classes derived from vcallable are required to align with their vcallable subobject",
				Classes_derived_from_vcallable_required_to_align_with_vcallable_subobject
			);
	}
	
	LEAN_INLINE vcallable(const vcallable &right) noexcept
		: m_fun(right.m_fun) { }
	
	LEAN_INLINE vcallable& operator =(const vcallable &right) noexcept
	{
		m_fun = right.m_fun;
		return *this;
	}

public:

#ifdef DOXYGEN_READ_THIS
	/// Calls the function stored by this callable object.
	LEAN_INLINE void operator ()(...)
	{
		(this->*m_fun)(...);
	}
#else
	#define LEAN_VCALLABLE_METHOD_DECL \
		LEAN_INLINE void operator ()
	#define LEAN_VCALLABLE_METHOD_BODY(call) \
		{ \
			(this->*m_fun)##call; \
		}
	LEAN_VARIADIC_TEMPLATE(LEAN_FORWARD, LEAN_VCALLABLE_METHOD_DECL, LEAN_NOTHING, LEAN_VCALLABLE_METHOD_BODY)
#endif
	
};

/// Stores a pointer to a method to be called on invokation of operator ().
template <class Signature, class Derived>
class vcallable_base : public vcallable<Signature>
{
protected:
	/// Stores the given object and method to be called by operator().
	LEAN_INLINE vcallable_base() noexcept
		: vcallable<Signature>( &Derived::operator () ) { }

	LEAN_INLINE vcallable_base(const vcallable_base &right) noexcept
		: vcallable<Signature>(right) { }
	
	LEAN_INLINE vcallable_base& operator =(const vcallable_base &right) noexcept
	{
		vcallable<Signature>::operator =(right);
		return *this;
	}
};

} // namespace

using functional::callable_fun;
using functional::callable_memfun;

using functional::make_callable;

using functional::vcallable;
using functional::vcallable_base;

} // namespace

#endif