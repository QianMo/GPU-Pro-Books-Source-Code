/*****************************************************/
/* lean built-in types          (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_TYPES
#define LEAN_TYPES

#include <cstddef>
#include <climits>
#include "cpp0x.h"

#ifdef DOXYGEN_READ_THIS
	/// Define this if sizeof(long) != sizeof(int).
	/// @ingroup GlobalSwitches
	#define LEAN_LONG_LONGER
	#undef LEAN_LONG_LONGER
	/// Define this if sizeof(short) == sizeof(int).
	/// @ingroup GlobalSwitches
	#define LEAN_INT_SHORTER
	#undef LEAN_INT_SHORTER
#endif

namespace lean
{

/// Defines fixed-width and other standard types.
namespace types
{

/// Sign classes enumeration.
struct sign_class
{
	/// Sign classes enumeration.
	enum t
	{
		no_sign,	///< Unsigned class.
		sign		///< Signed class.
	};
};

/// Provides an integer type of the given class and size.
template <sign_class::t Class, size_t Size>
struct int_type
{
	// Always checked, therefore use static_assert with care
	LEAN_STATIC_ASSERT_MSG_ALT(Size & ~Size, // = false, dependent
		"No integer type of the given size available.",
		No_integer_type_of_the_given_size_available);

	/// Integer type.
	typedef void type;
};

#ifndef DOXYGEN_SKIP_THIS

// Defaults that should work with most compilers
template<> struct int_type<sign_class::sign, sizeof(char)> { typedef char type; };
#ifndef LEAN_INT_SHORTER
template<> struct int_type<sign_class::sign, sizeof(short)> { typedef short type; };
#endif
template<> struct int_type<sign_class::sign, sizeof(int)> { typedef int type; };
#ifdef LEAN_LONG_LONGER
template<> struct int_type<sign_class::sign, sizeof(long)> { typedef long type; };
#endif
template<> struct int_type<sign_class::sign, sizeof(long long)> { typedef long long type; };

template<> struct int_type<sign_class::no_sign, sizeof(unsigned char)> { typedef unsigned char type; };
#ifndef LEAN_INT_SHORTER
template<> struct int_type<sign_class::no_sign, sizeof(unsigned short)> { typedef unsigned short type; };
#endif
template<> struct int_type<sign_class::no_sign, sizeof(unsigned int)> { typedef unsigned int type; };
#ifdef LEAN_LONG_LONGER
template<> struct int_type<sign_class::sign, sizeof(unsigned long)> { typedef unsigned long type; };
#endif
template<> struct int_type<sign_class::no_sign, sizeof(unsigned long long)> { typedef unsigned long long type; };

#endif

// Count bytes rather than bits (number of bits per char undefined)

/// 1 byte unsigned integer.
typedef int_type<sign_class::sign, 1>::type int1;
/// 2 byte unsigned integer.
typedef int_type<sign_class::sign, 2>::type int2;
/// 4 byte unsigned integer.
typedef int_type<sign_class::sign, 4>::type int4;
/// 8 byte unsigned integer.
typedef int_type<sign_class::sign, 8>::type int8;

/// 1 byte unsigned integer.
typedef int_type<sign_class::no_sign, 1>::type uint1;
/// 2 byte unsigned integer.
typedef int_type<sign_class::no_sign, 2>::type uint2;
/// 4 byte unsigned integer.
typedef int_type<sign_class::no_sign, 4>::type uint4;
/// 8 byte unsigned integer.
typedef int_type<sign_class::no_sign, 8>::type uint8;

/// Tristate.
typedef int1 tristate;
/// Require true.
static const tristate caretrue = true;
/// Require false.
static const tristate carefalse = false;
/// Don't care.
static const tristate dontcare = -1;

/// Provides a character type of the given size.
template <size_t Size>
struct char_type
{
	/// Character type.
	typedef typename int_type<sign_class::sign, Size>::type type;
};

#ifndef DOXYGEN_SKIP_THIS

// Defaults that should work with most compilers
template<> struct char_type<sizeof(char)> { typedef char type; };
template<> struct char_type<sizeof(wchar_t)> { typedef wchar_t type; };

#endif

/// Character type.
typedef char_type<1>::type char1;
/// Character type.
typedef char_type<2>::type char2;
/// Character type.
typedef char_type<4>::type char4;

/// Character type.
typedef char1 utf8_t;
/// Character type.
typedef char2 utf16_t;
/// Character type.
typedef char4 utf32_t;

/// Provides a floating-point type of the given size.
template <size_t Size>
struct float_type
{
	// Always checked, therefore use static_assert with care
	LEAN_STATIC_ASSERT_MSG_ALT(Size & ~Size, // = false, dependent
		"No floating-point type of the given size available.",
		No_floating_point_type_of_the_given_size_available);

	/// Floating-point type.
	typedef void type;
};

#ifndef DOXYGEN_SKIP_THIS

// Defaults that should work with most compilers
template<> struct float_type<sizeof(float)> { typedef float type; };
template<> struct float_type<sizeof(double)> { typedef double type; };

#endif

/// 4 byte float.
typedef float_type<4>::type float4;
/// 8 byte float.
typedef float_type<8>::type float8;

/// Describes the given float type.
template <class Float>
struct ieee_float_desc
{
	// Always checked, therefore use static_assert with care
	LEAN_STATIC_ASSERT_MSG_ALT(!sizeof(Float), // = false, dependent
		"No utility methods available for the given floating-point type.",
		No_utility_methods_available_for_the_given_floating_point_type);

	/// Float type.
	typedef Float float_type;
	/// Corresponding integer type.
	typedef typename types::int_type<sign_class::no_sign, sizeof(Float)>::type int_type;
	/// Corresponding decimal point shifting type.
	typedef typename types::int_type<sign_class::sign, sizeof(Float)>::type shift_type;

	/// Mantisssa bit count.
	static const size_t mantissa_bits;
	/// Exponent bit count.
	static const size_t exponent_bits;
};

#ifndef DOXYGEN_SKIP_THIS

template <>
struct ieee_float_desc<float4>
{
	typedef float4 float_type;
	typedef uint4 int_type;
	typedef int4 shift_type;

	static const size_t mantissa_bits = 23;
	static const size_t exponent_bits = 8;
	static const shift_type exponent_bias = 127;
};

template <>
struct ieee_float_desc<float8>
{
	typedef float8 float_type;
	typedef uint8 int_type;
	typedef int4 shift_type;

	static const size_t mantissa_bits = 52;
	static const size_t exponent_bits = 11;
	static const shift_type exponent_bias = 1023;
};

#endif

/// Provides utility methods for floating-point values of the given type.
template < class Float, class Desc = ieee_float_desc<Float> >
struct ieee_float : public Desc
{
	/// Desc type.
	typedef Desc desc_type;

	typedef typename Desc::float_type float_type;
	typedef typename Desc::int_type int_type;
	typedef typename Desc::shift_type shift_type;

	/// Mantissa bitmask.
	static const int_type mantissa_mask = (static_cast<int_type>(1) << mantissa_bits) - static_cast<int_type>(1);
	/// Exponent bitmask.
	static const int_type exponent_mask = (static_cast<int_type>(1) << exponent_bits) - static_cast<int_type>(1);

	/// Gets the (raw) mantissa of the given value.
	static int_type mantissa(float_type value)
	{
		return reinterpret_cast<const int_type&>(value) & mantissa_mask;
	}
	/// Gets the (raw) exponent of the given value.
	static int_type exponent(float_type value)
	{
		return (reinterpret_cast<const int_type&>(value) >> mantissa_bits) & exponent_mask;
	}
	/// Gets the (raw) sign of the given value (0 -> positive, 1 -> negative).
	static int_type sign(float_type value)
	{
		return reinterpret_cast<const int_type&>(value) >> (exponent_bits + mantissa_bits);
	}

	/// Gets the position of the decimal point.
	static shift_type shift(int_type exponent)
	{
		// Handle subnormal
		if (exponent == static_cast<int_type>(0))
			exponent = 1;
		
		return static_cast<shift_type>(exponent) - (exponent_bias + static_cast<shift_type>(mantissa_bits));
	}
	/// Gets the unshifted absolute value.
	static int_type fixed(int_type mantissa, int_type exponent)
	{
		// Add implicit one, if not subnormal
		if (exponent != static_cast<int_type>(0))
			mantissa |= (static_cast<int_type>(1) << mantissa_bits);

		return mantissa;
	}
	/// Gets whether the given value represents infinity.
	static bool is_special(int_type exponent)
	{
		return (exponent == exponent_mask);
	}
	/// Gets whether the given value represents infinity (provided is_special() returned true).
	static bool is_infinity(int_type mantissa)
	{
		return (mantissa == static_cast<int_type>(0));
	}
	/// Gets whether the given value represents NaN (provided is_special() returned true).
	static bool is_nan(int_type mantissa)
	{
		return (mantissa != static_cast<int_type>(0));
	}
};

/// Std size type.
using std::size_t;
/// Std pointer difference type.
using std::ptrdiff_t;

#ifdef LEAN0X_NO_UINTPTR_T
	/// Std pointer address type.
	typedef int_type<sign_class::no_sign, sizeof(void*)>::type uintptr_t;
#else
	/// Std pointer address type.
	using ::uintptr_t;
#endif

/// Number of bits per byte.
static const size_t bits_per_byte = CHAR_BIT;

/// Provides enhanced type size information.
template <class Type>
struct size_info
{
	/// Number of bytes.
	static const size_t bytes = sizeof(Type);
	/// Number of bits.
	static const size_t bits = bytes * bits_per_byte;
};

}

using types::sign_class;
using types::int_type;
using types::float_type;
using types::char_type;

using types::int1;
using types::int2;
using types::int4;
using types::int8;

using types::uint1;
using types::uint2;
using types::uint4;
using types::uint8;

using types::tristate;
using types::caretrue;
using types::carefalse;
using types::dontcare;

using types::char1;
using types::char2;
using types::char4;

using types::utf8_t;
using types::utf16_t;
using types::utf32_t;

using types::float4;
using types::float8;

using types::ieee_float_desc;
using types::ieee_float;

using types::size_t;
using types::ptrdiff_t;
using types::uintptr_t;

using types::bits_per_byte;
using types::size_info;

}

/// Redeclares the numeric types defined by lean in a namespace of the given name.
#define LEAN_REDECLARE_NUMERIC_TYPES(namespacename) namespace namespacename \
	{ \
		using ::lean::types::char1; \
		using ::lean::types::char2; \
		using ::lean::types::char4; \
		\
		using ::lean::types::int1; \
		using ::lean::types::int2; \
		using ::lean::types::int4; \
		using ::lean::types::int8; \
		\
		using ::lean::types::uint1; \
		using ::lean::types::uint2; \
		using ::lean::types::uint4; \
		using ::lean::types::uint8; \
		\
		using ::lean::types::float4; \
		using ::lean::types::float8; \
	}

/// Redeclares the numeric types defined by lean in a namespace of the given name.
#define LEAN_REIMPORT_NUMERIC_TYPES LEAN_REDECLARE_NUMERIC_TYPES(lean_numeric_types) using namespace lean_numeric_types;

#endif