/*****************************************************/
/* lean IO                      (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_IO_EDIANNESS
#define LEAN_IO_EDIANNESS

#include "../lean.h"

#ifdef _MSC_VER

#include <intrin.h>

namespace lean
{
namespace io
{
namespace impl
{

/// Swaps the byte order of the given value.
__forceinline unsigned short byteswap(unsigned short value)
{
	return _byteswap_ushort(value);
}

/// Swaps the byte order of the given value.
__forceinline unsigned long byteswap(unsigned long value)
{
	return _byteswap_ulong(value);
}

/// Swaps the byte order of the given value.
__forceinline unsigned long long byteswap(unsigned long long value)
{
	return _byteswap_uint64(value);
}

template <size_t Size>
struct swap_type
{
	// Always checked, therefore use static_assert with care
	LEAN_STATIC_ASSERT_MSG_ALT(Size & ~Size, // = false, dependent
		"Byte swap operations on values of the given type unsupported.",
		Byte_Swap_operations_on_values_of_the_given_type_unsupported);
};

template <> struct swap_type<sizeof(unsigned short)> { typedef unsigned short type; };
template <> struct swap_type<sizeof(unsigned long)> { typedef unsigned long type; };
template <> struct swap_type<sizeof(unsigned long long)> { typedef unsigned long long type; };

} // namespace
} // namespace
} // namespace

#else

#error Unknown compiler, intrinsics unavailable.

#endif

namespace lean
{
namespace io
{

/// Swaps the byte order of the given value.
template <class Value>
LEAN_INLINE Value byteswap(Value value)
{
	typedef typename impl::swap_type<sizeof(Value)>::type swap_int;

	return reinterpret_cast<const Value&>(static_cast<const swap_int&>( impl::byteswap(
			reinterpret_cast<const swap_int&>(value) ) ));
}

/// Swaps the byte order of the given values.
template <class Value>
LEAN_INLINE void byteswap(const Value *value, const Value *valueEnd, Value *dest)
{
	LEAN_ASSERT(dest);
	LEAN_ASSERT(value);
	LEAN_ASSERT(value <= valueEnd);

	while (value != valueEnd)
		*(dest++) = byteswap(*(value++));
}

/// Swaps the byte order of the given values.
template <class Value>
LEAN_INLINE void bytecopy(const Value *value, const Value *valueEnd, Value *dest)
{
	LEAN_ASSERT(dest);
	LEAN_ASSERT(value);
	LEAN_ASSERT(value <= valueEnd);

	memcpy(dest, value, reinterpret_cast<const char*>(valueEnd) - reinterpret_cast<const char*>(value));
}

#ifndef LEAN_BIG_ENDIAN

/// Sets the byte order of the given value to little endian.
template <class Value>
LEAN_INLINE Value byteswap_little(Value value)
{
	return value;
}

/// Sets the byte order of the given value to big endian.
template <class Value>
LEAN_INLINE Value byteswap_big(Value value)
{
	return byteswap(value);
}

/// Sets the byte order of the given values to little endian.
template <class Value>
LEAN_INLINE void byteswap_little(const Value *value, const Value *valueEnd, Value *dest)
{
	bytecopy(value, valueEnd, dest);
}

/// Sets the byte order of the given values to big endian.
template <class Value>
LEAN_INLINE void byteswap_big(const Value *value, const Value *valueEnd, Value *dest)
{
	byteswap(value, valueEnd, dest);
}

#else

/// Sets the byte order of the given value to little endian.
template <class Value>
LEAN_INLINE Value byteswap_little(Value value)
{
	return byteswap(value);
}

/// Sets the byte order of the given value to big endian.
template <class Value>
LEAN_INLINE Value byteswap_big(Value value)
{
	return value;
}

/// Sets the byte order of the given values to little endian.
template <class Value>
LEAN_INLINE void byteswap_little(const Value *value, const Value *valueEnd, Value *dest)
{
	byteswap(value, valueEnd, dest);
}

/// Sets the byte order of the given values to big endian.
template <class Value>
LEAN_INLINE void byteswap_big(const Value *value, const Value *valueEnd, Value *dest)
{
	bytecopy(value, valueEnd, dest);
}

#endif

} // namespace

using io::byteswap;
using io::byteswap_little;
using io::byteswap_big;

} // namespace

#endif