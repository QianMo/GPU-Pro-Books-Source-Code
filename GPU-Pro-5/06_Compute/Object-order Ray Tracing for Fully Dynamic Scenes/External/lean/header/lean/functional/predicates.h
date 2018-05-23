/*****************************************************/
/* lean Functional              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_FUNCTIONAL_PREDICATES
#define LEAN_FUNCTIONAL_PREDICATES

#include "../lean.h"

namespace lean
{
namespace functional
{

/// Returns whether the given number is even.
template <class Integer>
LEAN_INLINE bool is_even(Integer i)
{
	return ((i & Integer(1)) == Integer(0));
}

/// Returns whether the given number is odd.
template <class Integer>
LEAN_INLINE bool is_odd(Integer i)
{
	return !is_even(i);
}

/// Returns whether the given number is even.
template <size_t Divisor, class Integer>
LEAN_INLINE bool is_even(Integer i)
{
	return (i % Integer(Divisor) == Integer(0));
}

/// Returns whether the given number is odd.
template <size_t Divisor, class Integer>
LEAN_INLINE bool is_odd(Integer i)
{
	return !is_even<Divisor>(i);
}

} // namespace

using functional::is_even;
using functional::is_odd;

} // namespace

#endif