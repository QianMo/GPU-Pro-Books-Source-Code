/*****************************************************/
/* lean IO                      (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_IO_SERIALIZER
#define LEAN_IO_SERIALIZER

#include "../lean.h"
#include "../strings/types.h"
#include <typeinfo>
#include <iosfwd>

namespace lean
{
namespace io
{

/// Value serializer.
class serializer
{
	LEAN_INTERFACE_BEHAVIOR(serializer)

public:
	/// Gets the maximum length of the given number of values when serialized. Zero if unpredictable.
	virtual size_t max_length(size_t count) const = 0;

	/// Writes the given number of values to the given stream.
	virtual bool write(std::basic_ostream<utf8_t> &stream, const std::type_info &type, const void *values, size_t count) const = 0;
	/// Writes the given number of values to the given character buffer, returning the first character not written to.
	virtual utf8_t* write(utf8_t *begin, const std::type_info &type, const void *values, size_t count) const = 0;

	/// Reads the given number of values from the given stream.
	virtual bool read(std::basic_istream<utf8_t> &stream, const std::type_info &type, void *values, size_t count) const = 0;
	/// Reads the given number of values from the given range of characters, returning the first character not read.
	virtual const utf8_t* read(const utf8_t *begin, const utf8_t *end, const std::type_info &type, void *values, size_t count) const = 0;

	/// Gets the STD lib typeid.
	virtual const std::type_info& type_info() const = 0;
};

} // namespace

using io::serializer;

} // namespace

#endif