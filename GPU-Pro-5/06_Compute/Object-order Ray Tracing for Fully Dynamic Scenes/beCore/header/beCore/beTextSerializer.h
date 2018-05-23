/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_TEXT_SERIALIZER
#define BE_CORE_TEXT_SERIALIZER

#include "beCore.h"
#include <iosfwd>
#include <typeinfo>

namespace beCore
{

/// Text serializer.
class LEAN_INTERFACE TextSerializer
{
	LEAN_INTERFACE_BEHAVIOR(TextSerializer)

public:
	/// Gets the maximum length of the given number of values when serialized. Zero if unpredictable.
	virtual size_t GetMaxLength(size_t count) const = 0;

	/// Writes the given number of values to the given stream.
	virtual bool Write(std::basic_ostream<utf8_t> &stream, const std::type_info &type, const void *values, size_t count) const = 0;
	/// Writes the given number of values to the given character buffer, returning the first character not written to.
	virtual utf8_t* Write(utf8_t *begin, const std::type_info &type, const void *values, size_t count) const = 0;

	/// Reads the given number of values from the given stream.
	virtual bool Read(std::basic_istream<utf8_t> &stream, const std::type_info &type, void *values, size_t count) const = 0;
	/// Reads the given number of values from the given range of characters, returning the first character not read.
	virtual const utf8_t* Read(const utf8_t *begin, const utf8_t *end, const std::type_info &type, void *values, size_t count) const = 0;

	/// Gets the type name.
	virtual lean::utf8_ntr GetType() const = 0;
};

} // namespace

#endif