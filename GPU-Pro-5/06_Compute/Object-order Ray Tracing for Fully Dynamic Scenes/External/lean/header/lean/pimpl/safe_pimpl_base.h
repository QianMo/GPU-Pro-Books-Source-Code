/*****************************************************/
/* lean PImpl                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PIMPL_SAFE_PIMPL_BASE
#define LEAN_PIMPL_SAFE_PIMPL_BASE

#include "../lean.h"

namespace lean
{
namespace pimpl
{

/// Base class that permits safe destruction for incomplete private implementation classes.
class safe_pimpl_base
{
protected:
	LEAN_INLINE safe_pimpl_base() noexcept { }
	LEAN_INLINE safe_pimpl_base(const safe_pimpl_base&) noexcept { }
	LEAN_INLINE safe_pimpl_base& operator =(const safe_pimpl_base&) noexcept { return *this;}

public:
	/// Virtual destructor guarantees correct destruction,
	/// even where the actual pimpl class is unknown.
	virtual ~safe_pimpl_base() noexcept { };
};

} // namespace

using pimpl::safe_pimpl_base;

} // namespace

#endif