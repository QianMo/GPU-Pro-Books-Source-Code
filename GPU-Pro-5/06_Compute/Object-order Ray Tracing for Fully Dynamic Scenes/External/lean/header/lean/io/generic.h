/*****************************************************/
/* lean IO                      (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_IO_GENERIC
#define LEAN_IO_GENERIC

#include "../lean.h"
#include "../strings/types.h"
#include "serializer.h"
#include <typeinfo>
#include <iostream>
#include "../strings/charstream.h"
#include "../io/numeric.h"

namespace lean
{
namespace io
{

/// Generic value serialization.
template <class Type, utf8_t Delimiter = ';'>
class generic_serialization
{
public:
	/// Value type.
	typedef Type value_type;
	/// Delimiter character.
	static const utf8_t delimiter = Delimiter;

	/// Gets the maximum length of the given number of values when serialized. Zero if unpredictable.
	static size_t max_length(size_t count) { return 0; }

	// Writes the given number of values to the given stream.
	static bool write(std::basic_ostream<utf8_t> &stream, const std::type_info &type, const void *values, size_t count)
	{
		if (type != typeid(value_type))
			return false;

		const value_type *typedValues = static_cast<const value_type*>(values);
		
		for (size_t i = 0; i < count; ++i)
		{
			if (i != 0)
				stream << delimiter;

			stream << typedValues[i];
		}

		return !stream.fail();
	}

	// Writes the given number of values to the given character buffer, returning the first character not written to.
	static utf8_t* write(utf8_t *begin, const std::type_info &type, const void *values, size_t count)
	{
		basic_charstream<utf8_t> stream(begin);
		write(stream, type, values, count);
		return stream.write_end();
	}

	// Reads the given number of values from the given stream.
	static bool read(std::basic_istream<utf8_t> &stream, const std::type_info &type, void *values, size_t count)
	{
		if (type != typeid(value_type))
			return false;

		value_type *typedValues = static_cast<value_type*>(values);
		
		for (size_t i = 0; i < count; ++i)
		{
			if (i != 0)
				stream.ignore(std::numeric_limits<int>::max(), delimiter); // required to be int

			stream >> typedValues[i];
		}

		return !stream.fail();
	}

	// Reads the given number of values from the given range of characters, returning the first character not read.
	static const utf8_t* read(const utf8_t *begin, const utf8_t *end, const std::type_info &type, void *values, size_t count)
	{
		basic_charstream<utf8_t> stream(const_cast<utf8_t*>(begin), const_cast<utf8_t*>(end));
		read(stream, type, values, count);
		return stream.read_end();
	}
};

/// Boolean value serialization.
template <class Type, utf8_t Delimiter = ';'>
struct bool_serialization : public generic_serialization<Type, Delimiter>
{
	/// Value type.
	typedef Type value_type;
	/// Delimiter character.
	static const utf8_t delimiter = Delimiter;

	/// Gets the maximum length of the given number of values when serialized. Zero if unpredictable.
	static size_t max_length(size_t count)
	{
		// N numbers & delimiters
		return (1 + 1) * count;
	}

	// Writes the given number of values to the given character buffer, returning the first character not written to.
	static utf8_t* write(utf8_t *begin, const std::type_info &type, const void *values, size_t count)
	{
		if (type != typeid(value_type))
			return begin;

		const value_type *typedValues = static_cast<const value_type*>(values);
		
		for (size_t i = 0; i < count; ++i)
		{
			if (i != 0)
				*begin++ = delimiter;

			begin = bool_to_char(begin, typedValues[i]);
		}

		return begin;
	}
	using generic_serialization<Type, Delimiter>::write;

	// Reads the given number of values from the given range of characters, returning the first character not read.
	static const utf8_t* read(const utf8_t *begin, const utf8_t *end, const std::type_info &type, void *values, size_t count)
	{
		if (type != typeid(value_type))
			return begin;

		value_type *typedValues = static_cast<value_type*>(values);

		for (size_t i = 0; i < count; ++i)
		{
			if (i != 0)
				// Skip UNTIL next delimiter found
				while (begin != end && *begin++ != delimiter);
			
			// Skip white space
			while (begin != end && *begin == ' ')
				begin++;

			begin = char_to_bool(begin, end, typedValues[i]);
		}

		return begin;
	}
	using generic_serialization<Type, Delimiter>::read;
};

/// Integer value serialization.
template <class Type, utf8_t Delimiter = ';'>
struct int_serialization : public generic_serialization<Type, Delimiter>
{
	/// Value type.
	typedef Type value_type;
	/// Delimiter character.
	static const utf8_t delimiter = Delimiter;

	/// Gets the maximum length of the given number of values when serialized. Zero if unpredictable.
	static size_t max_length(size_t count)
	{
		// N numbers & delimiters
		return (max_int_string_length<Type>::value + 1) * count;
	}

	// Writes the given number of values to the given character buffer, returning the first character not written to.
	static utf8_t* write(utf8_t *begin, const std::type_info &type, const void *values, size_t count)
	{
		if (type != typeid(value_type))
			return begin;

		const value_type *typedValues = static_cast<const value_type*>(values);
		
		for (size_t i = 0; i < count; ++i)
		{
			if (i != 0)
				*begin++ = delimiter;

			begin = int_to_char(begin, typedValues[i]);
		}

		return begin;
	}
	using generic_serialization<Type, Delimiter>::write;

	// Reads the given number of values from the given range of characters, returning the first character not read.
	static const utf8_t* read(const utf8_t *begin, const utf8_t *end, const std::type_info &type, void *values, size_t count)
	{
		if (type != typeid(value_type))
			return begin;

		value_type *typedValues = static_cast<value_type*>(values);

		for (size_t i = 0; i < count; ++i)
		{
			if (i != 0)
				// Skip UNTIL next delimiter found
				while (begin != end && *begin++ != delimiter);

			// Skip white space
			while (begin != end && *begin == ' ')
				begin++;

			begin = char_to_int(begin, end, typedValues[i]);
		}

		return begin;
	}
	using generic_serialization<Type, Delimiter>::read;
};

/// Float value serialization.
template <class Type, utf8_t Delimiter = ';'>
struct float_serialization : public generic_serialization<Type, Delimiter>
{
	/// Value type.
	typedef Type value_type;
	/// Delimiter character.
	static const utf8_t delimiter = Delimiter;

	/// Gets the maximum length of the given number of values when serialized. Zero if unpredictable.
	static size_t max_length(size_t count)
	{
		// N numbers & delimiters
		return (max_float_string_length<Type>::value + 1) * count;
	}

	// Writes the given number of values to the given character buffer, returning the first character not written to.
	static utf8_t* write(utf8_t *begin, const std::type_info &type, const void *values, size_t count)
	{
		if (type != typeid(value_type))
			return begin;

		const value_type *typedValues = static_cast<const value_type*>(values);
		
		for (size_t i = 0; i < count; ++i)
		{
			if (i != 0)
				*begin++ = delimiter;

			begin = float_to_char(begin, typedValues[i]);
		}

		return begin;
	}
	using generic_serialization<Type, Delimiter>::write;

	// Reads the given number of values from the given range of characters, returning the first character not read.
	static const utf8_t* read(const utf8_t *begin, const utf8_t *end, const std::type_info &type, void *values, size_t count)
	{
		if (type != typeid(value_type))
			return begin;

		value_type *typedValues = static_cast<value_type*>(values);

		for (size_t i = 0; i < count; ++i)
		{
			if (i != 0)
				// Skip UNTIL next delimiter found
				while (begin != end && *begin++ != delimiter);

			// Skip white space
			while (begin != end && *begin == ' ')
				begin++;

			begin = char_to_float(begin, end, typedValues[i]);
		}

		return begin;
	}
	using generic_serialization<Type, Delimiter>::read;
};

/// Generic value serializer.
template <class Serialization>
class generic_serializer : public serializer, public Serialization
{
public:
	/// Serialization type.
	typedef Serialization serialization_type;

	/// Gets the maximum length of the given number of values when serialized. Zero if unpredictable.
	size_t max_length(size_t count) const { return serialization_type::max_length(); }

	// Writes the given number of values to the given stream.
	bool write(std::basic_ostream<utf8_t> &stream, const std::type_info &type, const void *values, size_t count) const { return serialization_type::write(stream, type, values, count); }
	// Writes the given number of values to the given character buffer, returning the first character not written to.
	utf8_t* write(utf8_t *begin, const std::type_info &type, const void *values, size_t count) const { return serialization_type::write(begin, type, values, count); }

	// Reads the given number of values from the given stream.
	bool read(std::basic_istream<utf8_t> &stream, const std::type_info &type, void *values, size_t count) const { return serialization_type::read(stream, type, values, count); }
	// Reads the given number of values from the given range of characters, returning the first character not read.
	const utf8_t* read(const utf8_t *begin, const utf8_t *end, const std::type_info &type, void *values, size_t count) const { return serialization_type::read(begin, end, type, values, count); }

	/// Gets the STD lib typeid.
	const std::type_info& type_info() const { return typeid(Type); }
};

/// Gets the generic serializer for the given type.
template <class Type>
LEAN_NOINLINE const serializer& get_generic_serializer()
{
	static generic_serializer< generic_serialization<Type> > serializer;
	return serializer;
}

/// Gets the integer serializer for the given type.
template <class Type>
LEAN_NOINLINE const serializer& get_int_serializer()
{
	static generic_serializer< int_serialization<Type> > serializer;
	return serializer;
}

/// Gets the float serializer for the given type.
template <class Type>
LEAN_NOINLINE const serializer& get_float_serializer()
{
	static generic_serializer< float_serialization<Type> > serializer;
	return serializer;
}

} // namespace

using io::generic_serialization;
using io::int_serialization;
using io::float_serialization;

using io::get_generic_serializer;
using io::get_int_serializer;
using io::get_float_serializer;

} // namespace

#endif