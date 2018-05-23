/*****************************************************/
/* lean Memory                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MEMORY
#define LEAN_MEMORY

namespace lean
{
	/// Provides a flexible heap concept, memory alignment facilities and a comprehensive way of handling out-of-memory situations.
	namespace memory { }
}

#include "new_handler.h"

#include "alignment.h"
#include "crt_heap.h"
#include "default_heap.h"

#include "heap_allocator.h"

#include "heap_bound.h"
#include "aligned.h"

/// @defgroup MemorySwitches Memory-management switches

#endif