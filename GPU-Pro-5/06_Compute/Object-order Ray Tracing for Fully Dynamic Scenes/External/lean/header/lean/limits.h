/*****************************************************/
/* lean built-in types          (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_LIMITS
#define LEAN_LIMITS

#include <cstddef>
#include <limits>
#include <cfloat>
#include "cpp0x.h"

#ifdef DOXYGEN_READ_THIS
	/// Define this to enable distinct size_t specializations.
	/// @ingroup GlobalSwitches
	#define LEAN_BUILTIN_SIZE_T
	#undef LEAN_BUILTIN_SIZE_T
#endif

namespace lean
{

namespace types
{

namespace
{

/// Provides literal numeric limits information.
template <class Type>
struct numeric_limits
{
	// Always checked, therefore use static_assert with care
	LEAN_STATIC_ASSERT_MSG_ALT(!sizeof(Type), // = false, dependent
		"No numeric limits available for the given type.",
		No_numeric_limits_available_for_the_given_type);
	
	/// Specifies whether the type is an integer type.
	static const bool is_int;
	/// Specifies whether the type is an floating-point type.
	static const bool is_float;
	/// Specifies whether the type is unsigned.
	static const bool is_unsigned;
	/// Smallest value.
	static const Type min;
	/// Greatest value.
	static const Type max;
};

namespace limits_impl
{
	template <class Integer>
	struct int_limits_base
	{
		static const bool is_int = true;
		static const bool is_float = false;
	};

	template <class SInteger>
	struct sint_limits_base : public int_limits_base<SInteger>
	{
		static const bool is_unsigned = false;
	};

	template <class UInteger>
	struct uint_limits_base : public int_limits_base<UInteger>
	{
		static const bool is_unsigned = true;
		static const UInteger min = static_cast<UInteger>(0);
	};

	struct float_limits_base
	{
		static const bool is_int = false;
		static const bool is_float = true;
		static const bool is_unsigned = false;
	};

} // namespace

template <>
struct numeric_limits<char> : public limits_impl::sint_limits_base<char>
{
	static const char min = CHAR_MIN;
	static const char max = CHAR_MAX;
};
template <>
struct numeric_limits<unsigned char> : public limits_impl::uint_limits_base<unsigned char>
{
	static const unsigned char max = UCHAR_MAX;
};

template <>
struct numeric_limits<short> : public limits_impl::sint_limits_base<short>
{
	static const short min = SHRT_MIN;
	static const short max = SHRT_MAX;
};
template <>
struct numeric_limits<unsigned short> : public limits_impl::uint_limits_base<unsigned short>
{
	static const unsigned short max = USHRT_MAX;
};

template <>
struct numeric_limits<int> : public limits_impl::sint_limits_base<int>
{
	static const int min = INT_MIN;
	static const int max = INT_MAX;
};
template <>
struct numeric_limits<unsigned int> : public limits_impl::uint_limits_base<unsigned int>
{
	static const unsigned int max = UINT_MAX;
};

template <>
struct numeric_limits<long> : public limits_impl::sint_limits_base<long>
{
	static const long min = LONG_MIN;
	static const long max = LONG_MAX;
};
template <>
struct numeric_limits<unsigned long> : public limits_impl::uint_limits_base<unsigned long>
{
	static const unsigned long max = ULONG_MAX;
};

template <>
struct numeric_limits<long long> : public limits_impl::sint_limits_base<long long>
{
	static const long long min = LLONG_MIN;
	static const long long max = LLONG_MAX;
};
template <>
struct numeric_limits<unsigned long long> : public limits_impl::uint_limits_base<unsigned long long>
{
	static const unsigned long long max = ULLONG_MAX;
};

#ifdef LEAN_BUILTIN_SIZE_T
template <>
struct numeric_limits<size_t> : public limits_impl::uint_limits_base<size_t>
{
	static const size_t max = SIZE_MAX;
};
#endif

template <>
struct numeric_limits<float> : public limits_impl::float_limits_base
{
	static const float min;
	static const float max;
};
const float numeric_limits<float>::min = -FLT_MAX;
const float numeric_limits<float>::max = FLT_MAX;

template <>
struct numeric_limits<double> : public limits_impl::float_limits_base
{
	static const double min;
	static const double max;
};
const double numeric_limits<double>::min = -DBL_MAX;
const double numeric_limits<double>::max = DBL_MAX;

template <>
struct numeric_limits<long double> : public limits_impl::float_limits_base
{
	static const long double min;
	static const long double max;
};
const long double numeric_limits<long double>::min = -LDBL_MAX;
const long double numeric_limits<long double>::max = LDBL_MAX;

} // namespace

} // namespace

using types::numeric_limits;

} // namespace

#endif