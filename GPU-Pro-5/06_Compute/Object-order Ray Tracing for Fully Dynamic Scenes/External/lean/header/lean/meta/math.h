/*****************************************************/
/* lean Meta                    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_META_MATH
#define LEAN_META_MATH

#include "type.h"

namespace lean
{
namespace meta
{

template <class T>
struct zero
{
	static const T value = T(0);
};

template <class T, T B, T V>
struct log
{
	LEAN_STATIC_ASSERT_MSG_ALT(
			V > T(0),
			"Log undefined for values <= 0",
			Log_undefined_for_values_less_equal_zero
		);
	LEAN_STATIC_ASSERT_MSG_ALT(
			B > T(1),
			"Log undefined for bases <= 1",
			Log_undefined_for_bases_less_equal_one
		);

	static const T value = (V >= B)
		? T(1) + conditional_type< V >= B, log<T, B, V / B>, zero<T> >::type::value
		: T(0);
};

template <class T, T V>
struct log2
{
	static const T value = log<T, 2, V>::value;
};

template <class T, T V>
struct log10
{
	static const T value = log<T, 10, V>::value;
};

} // namespace

using meta::log;
using meta::log2;
using meta::log10;

} // namespace

#endif