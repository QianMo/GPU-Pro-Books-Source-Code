/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_GENERIC_TEXT_SERIALIZER
#define BE_CORE_GENERIC_TEXT_SERIALIZER

#include "beCore.h"
#include "beTextSerializer.h"
#include <lean/io/generic.h>

namespace beCore
{

/// Generic text serializer.
template <class LeanSerialization>
class GenericTextSerializer : public TextSerializer
{
public:
	/// Serialization type.
	typedef LeanSerialization serialization_type;

	/// Gets the maximum length of the given number of values when serialized. Zero if unpredictable.
	size_t GetMaxLength(size_t count) const { return serialization_type::max_length(count); }

	/// Writes the given number of values to the given stream.
	bool Write(std::basic_ostream<utf8_t> &stream, const std::type_info &type, const void *values, size_t count) const { return serialization_type::write(stream, type, values, count); }
	/// Writes the given number of values to the given character buffer, returning the first character not written to.
	utf8_t* Write(utf8_t *begin, const std::type_info &type, const void *values, size_t count) const { return serialization_type::write(begin, type, values, count); }

	/// Reads the given number of values from the given stream.
	bool Read(std::basic_istream<utf8_t> &stream, const std::type_info &type, void *values, size_t count) const { return serialization_type::read(stream, type, values, count); }
	/// Reads the given number of values from the given range of characters, returning the first character not read.
	const utf8_t* Read(const utf8_t *begin, const utf8_t *end, const std::type_info &type, void *values, size_t count) const { return serialization_type::read(begin, end, type, values, count); }

	/// Gets the type name.
	lean::utf8_ntr GetType() const { return lean::utf8_ntr(typeid(serialization_type::value_type).name()); }
};

} // namespace

#endif