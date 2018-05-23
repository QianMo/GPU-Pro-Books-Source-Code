/*****************************************************/
/* lean PImpl                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PIMPL_STATIC_PIMPL
#define LEAN_PIMPL_STATIC_PIMPL

#include "../lean.h"

/// @addtogroup PImplMacros PImpl macros
/// @see lean::pimpl
/// @{

/// Makes the given class behave like an interface to a static private implementation.
#define LEAN_SIMPL_INTERFACE_BEHAVIOR(name) LEAN_INTERFACE_BEHAVIOR(name) \
		private: \
			LEAN_INLINE name() noexcept { } \
			LEAN_INLINE name(const name&) noexcept { }

/// Makes the given class behave like an interface to a static private implementation, supporting shared ownership.
#define LEAN_SHARED_SIMPL_INTERFACE_BEHAVIOR(name) LEAN_SHARED_INTERFACE_BEHAVIOR(name) \
		private: \
			LEAN_INLINE name() noexcept { } \
			LEAN_INLINE name(const name&) noexcept { }

/// Defines a local reference to the private implementation of the given type and name.
#define LEAN_STATIC_NAMED_PIMPL_AT(t, m, w) t &m = static_cast<t&>(w)
/// Defines a local reference to the private implementation 'm' of type 'M'.
#define LEAN_STATIC_PIMPL_AT(w) LEAN_STATIC_NAMED_PIMPL_AT(M, m, w)
/// Defines a local reference to the private implementation 'm' of type 'M'.
#define LEAN_STATIC_PIMPL_AT_CONST(w) LEAN_STATIC_NAMED_PIMPL_AT(const M, m, w)

/// Defines a local reference to the private implementation of the given type and name.
#define LEAN_STATIC_NAMED_PIMPL(t, m) LEAN_STATIC_NAMED_PIMPL_AT(t, m, *this)
/// Defines a local reference to the private implementation 'm' of type 'M'.
#define LEAN_STATIC_PIMPL() LEAN_STATIC_PIMPL_AT(*this)
/// Defines a local reference to the private implementation 'm' of type 'M'.
#define LEAN_STATIC_PIMPL_CONST() LEAN_STATIC_PIMPL_AT_CONST(*this)

/// Defines a local type 'M' for the private implementation of type 'M'.
#define LEAN_FREE_PIMPL(t) typedef t::M M 
/// Defines a local reference to the private implementation 'm' of type 'M'.
#define LEAN_FREE_STATIC_PIMPL_AT(t, w) LEAN_FREE_PIMPL(t); LEAN_STATIC_PIMPL_AT(w)
/// Defines a local reference to the private implementation 'm' of type 'M'.
#define LEAN_FREE_STATIC_PIMPL_AT_CONST(t, w) LEAN_FREE_PIMPL(t); LEAN_STATIC_PIMPL_AT_CONST(w)

/// @}

#endif