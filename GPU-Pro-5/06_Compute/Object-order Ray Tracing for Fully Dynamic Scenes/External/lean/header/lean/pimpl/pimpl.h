/*****************************************************/
/* lean PImpl                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PIMPL_PIMPL
#define LEAN_PIMPL_PIMPL

namespace lean
{
	/// Defines classes that allow for the decoupling of modules, especially when it comes to the hiding of private dependencies.
	/// @see PImplMacros
	namespace pimpl { }
}

#include "safe_pimpl_base.h"
#include "unsafe_pimpl_base.h"
#include "pimpl_ptr.h"
#include "opaque_val.h"

#endif