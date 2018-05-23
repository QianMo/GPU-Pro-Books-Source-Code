/*****************************************************/
/* lean Memory                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MEMORY_NEW_HANDLER
#define LEAN_MEMORY_NEW_HANDLER

#include "../lean.h"

namespace lean
{
namespace memory
{

/// Lean new handler, returns true to request retry, false to abort allocation.
typedef bool (*new_handler)();

/// Sets a new new_handler.
LEAN_MAYBE_EXPORT new_handler set_new_handler(new_handler newHandler);

/// Calls the current new handler.
LEAN_MAYBE_EXPORT bool call_new_handler();

} // namespace

using memory::new_handler;
using memory::set_new_handler;
using memory::call_new_handler;

} // namespace

#ifdef LEAN_INCLUDE_LINKED
#include "source/new_handler.cpp"
#endif

#endif