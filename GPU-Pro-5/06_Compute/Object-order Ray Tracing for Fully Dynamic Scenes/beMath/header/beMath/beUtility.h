/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_UTILITY
#define BE_MATH_UTILITY

#include "beMath.h"
#include <lean/functional/variadic.h>

namespace beMath
{

/// -1 or 1, depending on the sign.
template <class Scalar>
LEAN_INLINE Scalar sign(Scalar s)
{
	return (s >= Scalar(0)) ? Scalar(1) : Scalar(-1);
}

/// -1, 0 or 1, depending on the value.
template <class Scalar>
LEAN_INLINE Scalar sign0(Scalar s)
{
	if (s > Scalar(0))
		return Scalar(1);
	else if (s < Scalar(0))
		return Scalar(-1);
	else
		return Scalar(0);
}

/// Value squared.
template <class Scalar>
LEAN_INLINE Scalar square(Scalar s)
{
	return s * s;
}

/// Gets the minimum of the given values.
template <class Scalar>
LEAN_INLINE Scalar min(Scalar a, Scalar b)
{
	return (a < b) ? a : b;
}
template <class Scalar>
LEAN_INLINE Scalar min(Scalar a, Scalar b, Scalar c)
{
	return min(min(a, b), c);
}
template <class Scalar>
LEAN_INLINE Scalar min(Scalar a, Scalar b, Scalar c, Scalar d)
{
	return min(min(a, b), min(c, d));
}

/// Gets the maximum of the given values.
template <class Scalar>
LEAN_INLINE Scalar max(Scalar a, Scalar b)
{
	return (a < b) ? b : a;
}
template <class Scalar>
LEAN_INLINE Scalar max(Scalar a, Scalar b, Scalar c)
{
	return max(max(a, b), c);
}
template <class Scalar>
LEAN_INLINE Scalar max(Scalar a, Scalar b, Scalar c, Scalar d)
{
	return max(max(a, b), max(c, d));
}

/// Clamps the given value to the given range.
template <class Scalar>
LEAN_INLINE Scalar clamp(Scalar s, Scalar a, Scalar b)
{
	return (a <= s) ? ((s <= b) ? s : b) : a;
}

template <class Scalar>
struct min_and_max_solution
{
	Scalar min;
	Scalar max;
};

/// Solves a x^2 + b x + c = 0.
template <class Scalar>
inline min_and_max_solution<Scalar> solve_sq(Scalar a, Scalar b, Scalar c)
{
	min_and_max_solution<Scalar> s;

	Scalar d = sqrt(b * b - Scalar(4) * a * c);
	Scalar dn = a + a;

	s.min = (-b - d) / dn;
	s.max = (-b + d) / dn;

	return s;
}

} // namespace

#endif