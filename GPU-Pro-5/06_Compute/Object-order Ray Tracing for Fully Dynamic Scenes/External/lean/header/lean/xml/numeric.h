/*****************************************************/
/* lean XML                     (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_XML_NUMERIC
#define LEAN_XML_NUMERIC

#include "../lean.h"
#include "../io/numeric.h"
#include "utility.h"

namespace lean
{
namespace xml
{

/// Appends an integer XML attribute to the given node.
template <class Char, class Integer, class Range>
inline void append_int_attribute(rapidxml::xml_document<Char> &document, rapidxml::xml_node<Char> &node, const Range &name, const Integer &value)
{
	Char numBuffer[max_int_string_length<Integer>::value + 1];
	
	Char *numEnd = int_to_char(numBuffer, value);
	*numEnd = 0;

	append_attribute( document, node, name, nullterminated_range<Char>(numBuffer, numEnd) );
}

/// Appends a floating-point XML attribute to the given node.
template <class Char, class Float, class Range>
inline void append_float_attribute(rapidxml::xml_document<Char> &document, rapidxml::xml_node<Char> &node, const Range &name, const Float &value)
{
	Char numBuffer[max_float_string_length<Float>::value + 1];
	
	Char *numEnd = float_to_char(numBuffer, value);
	*numEnd = 0;

	append_attribute( document, node, name, nullterminated_range<Char>(numBuffer, numEnd) );
}

/// Gets the integer value of the given XML attribute, returns the default value if not available.
template <class Char, class Integer, class Range>
inline Integer get_int_attribute(const rapidxml::xml_node<Char> &node, const Range &name, const Integer &defaultValue)
{
	Integer value(defaultValue);
	string_to_int( get_attribute(node, name), value );
	return value;
}

/// Gets the flooating-point value of the given XML attribute, returns the default value if not available.
template <class Char, class Float, class Range>
inline Float get_float_attribute(const rapidxml::xml_node<Char> &node, const Range &name, const Float &defaultValue)
{
	Float value(defaultValue);
	string_to_float( get_attribute(node, name), value );
	return value;
}

/// Gets the integer value of the given XML attribute, returns the default value if not available.
template <class Char, class Boolean, class Range>
inline Boolean get_bool_attribute(const rapidxml::xml_node<Char> &node, const Range &name, const Boolean &defaultValue)
{
	Boolean value(defaultValue);
	string_to_bool( get_attribute(node, name), value );
	return value;
}

/// Gets the minimum of the given integer attribute considering all direct child nodes.
template <class Int, class Char, class Range>
LEAN_INLINE Int min_int_attribute(const rapidxml::xml_node<Char> &node, const Range &attribute, Int startValue,
	const Char *name = nullptr, size_t nameSize = 0, bool caseSensitive = true)
{
	Int value = startValue;

	for (const rapidxml::xml_node<Char> *pNode = node.first_node(name, nameSize, caseSensitive);
		pNode; pNode = pNode->next_sibling(name, nameSize, caseSensitive))
		value = min( get_int_attribute(*pNode, attribute, value), value );

	return value;
}

/// Gets the maximum of the given integer attribute considering all direct child nodes.
template <class Int, class Char, class Range>
LEAN_INLINE Int max_int_attribute(const rapidxml::xml_node<Char> &node, const Range &attribute, Int startValue,
	const Char *name = nullptr, size_t nameSize = 0, bool caseSensitive = true)
{
	Int value = startValue;

	for (const rapidxml::xml_node<Char> *pNode = node.first_node(name, nameSize, caseSensitive);
		pNode; pNode = pNode->next_sibling(name, nameSize, caseSensitive))
		value = max( get_int_attribute(*pNode, attribute, value), value );

	return value;
}

} // namespace

using xml::append_int_attribute;
using xml::append_float_attribute;
using xml::get_int_attribute;
using xml::get_float_attribute;
using xml::get_bool_attribute;
using xml::min_int_attribute;
using xml::max_int_attribute;

} // namespace

#endif