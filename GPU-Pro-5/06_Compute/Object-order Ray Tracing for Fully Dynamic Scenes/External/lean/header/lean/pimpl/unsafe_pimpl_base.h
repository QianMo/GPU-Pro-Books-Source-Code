/*****************************************************/
/* lean PImpl                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PIMPL_UNSAFE_PIMPL_BASE
#define LEAN_PIMPL_UNSAFE_PIMPL_BASE

#include "../lean.h"

namespace lean
{
namespace pimpl
{

/// Base class that permits unsafe destruction for incomplete private implementation classes, ONLY USE IF CORRECT DESTRUCTION NOT REQUIRED.
class unsafe_pimpl_base
{
protected:
	LEAN_INLINE unsafe_pimpl_base() { }
	LEAN_INLINE unsafe_pimpl_base(const unsafe_pimpl_base&) { }
	LEAN_INLINE unsafe_pimpl_base& operator =(const unsafe_pimpl_base&) {  return *this;}
};

} // namespace

using pimpl::unsafe_pimpl_base;

} // namespace

#endif