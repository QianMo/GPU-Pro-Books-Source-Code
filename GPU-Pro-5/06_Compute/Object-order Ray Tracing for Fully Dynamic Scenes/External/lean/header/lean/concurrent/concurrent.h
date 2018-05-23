/*****************************************************/
/* lean Concurrent              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONCURRENT_CONCURRENT
#define LEAN_CONCURRENT_CONCURRENT

namespace lean
{
	/// Defines classes and utilities for multi-threaded environments such as spin locks, atomic operations, etc.
	namespace concurrent { }
}

#include "atomic.h"

#include "spin_lock.h"
#include "shareable_spin_lock.h"

#include "shareable_lock_policies.h"

#endif