/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_WRAPPER
#define BE_CORE_WRAPPER

#include "beCore.h"

namespace beCore
{

/// Base class for interface wrappers.
template <class Interface, class Derived>
class Wrapper
{
	LEAN_STATIC_INTERFACE_BEHAVIOR(Wrapper)
	
public:
	/// Gets the wrapped interface.
	LEAN_INLINE Interface*const& Get() { return static_cast<Derived&>(*this).GetInterface(); }
	
	/// Gets the wrapped interface.
	LEAN_INLINE Interface& operator *() { return *Get(); };
	
	/// Gets the wrapped interface.
	LEAN_INLINE Interface* operator ->() { return Get(); };
	
	/// Gets the wrapped interface.
	LEAN_INLINE operator Interface*() { return Get(); };
};

/// Base class for interface wrappers.
template <class Interface, class Derived>
class TransitiveWrapper
{
	LEAN_STATIC_INTERFACE_BEHAVIOR(TransitiveWrapper)
	
public:
	/// Gets the wrapped interface.
	LEAN_INLINE Interface*const& Get() { return static_cast<Derived&>(*this).GetInterface(); }
	/// Gets the wrapped interface.
	LEAN_INLINE const Interface*const& Get() const { return static_cast<const Derived&>(*this).GetInterface(); }
	
	/// Gets the wrapped interface.
	LEAN_INLINE Interface& operator *() { return *Get(); };
	/// Gets the wrapped interface.
	LEAN_INLINE const Interface& operator *() const { return *Get(); };
	
	/// Gets the wrapped interface.
	LEAN_INLINE Interface* operator ->() { return Get(); };
	/// Gets the wrapped interface.
	LEAN_INLINE const Interface* operator ->() const { return Get(); };

	/// Gets the wrapped interface.
	LEAN_INLINE operator Interface*() { return Get(); };
	/// Gets the wrapped interface.
	LEAN_INLINE operator const Interface*() const { return Get(); };
};

/// Base class for interface wrappers.
template <class Interface, class Derived>
class IntransitiveWrapper
{
	LEAN_STATIC_INTERFACE_BEHAVIOR(IntransitiveWrapper)
	
public:
	/// Gets the wrapped interface.
	LEAN_INLINE Interface*const& Get() const { return static_cast<const Derived&>(*this).GetInterface(); }
	
	/// Gets the wrapped interface.
	LEAN_INLINE Interface& operator *() const { return *Get(); };
	
	/// Gets the wrapped interface.
	LEAN_INLINE Interface* operator ->() const { return Get(); };
	
	/// Gets the wrapped interface.
	LEAN_INLINE operator Interface*() const { return Get(); };
};

} // namespace

#endif