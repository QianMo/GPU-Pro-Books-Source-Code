/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_COMPARISON
#define BE_MATH_COMPARISON

#include "beMath.h"

namespace beMath
{

/// Compares the two values given.
template <class T1, class T2>
LEAN_INLINE bool operator !=(const T1 &left, const T2 &right)
{
	return !(left == right);
}

/// Compares the two values given.
template <class T1, class T2>
LEAN_INLINE bool operator <=(const T1 &left, const T2 &right)
{
	return !(right < left);
}

/// Compares the two values given.
template <class T1, class T2>
LEAN_INLINE bool operator >=(const T1 &left, const T2 &right)
{
	return !(left < right);
}

/// Compares the two values given.
template <class T1, class T2>
LEAN_INLINE bool operator >(const T1 &left, const T2 &right)
{
	return (right < left);
}

} // namespace

#endif