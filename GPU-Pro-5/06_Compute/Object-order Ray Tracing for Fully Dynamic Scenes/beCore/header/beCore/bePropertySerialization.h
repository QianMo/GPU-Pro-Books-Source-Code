/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_PROPERTY_SERIALIZATION
#define BE_CORE_PROPERTY_SERIALIZATION

#include "beCore.h"
#include "bePropertyProvider.h"
#include "bePropertyVisitor.h"

#include <lean/rapidxml/rapidxml.hpp>

namespace beCore
{

/// Property serializer.
struct PropertySerializer : public PropertyVisitor
{
	const PropertyProvider *Properties;
	rapidxml::xml_node<lean::utf8_t> *Parent;
	bool IncludeType;

	/// Constructor.
	PropertySerializer(const PropertyProvider &Properties, rapidxml::xml_node<lean::utf8_t> &Parent, bool IncludeType = false)
		: Properties(&Properties),
		Parent(&Parent),
		IncludeType(IncludeType) { }

	/// Visits the given values.
	BE_CORE_API void Visit(const PropertyProvider &provider, uint4 propertyID, const PropertyDesc &desc, const void *values) LEAN_OVERRIDE;
};

/// Property deserializer.
struct PropertyDeserializer : public PropertyVisitor
{
	PropertyProvider *Properties;
	const rapidxml::xml_node<lean::utf8_t> *Node;

	/// Constructor.
	PropertyDeserializer(PropertyProvider &Properties, const rapidxml::xml_node<lean::utf8_t> &Node)
		: Properties(&Properties),
		Node(&Node) { }

	/// Gets the property name.
	BE_CORE_API utf8_ntr Name() const;
	/// Gets the number of values.
	BE_CORE_API utf8_ntr ValueType() const;
	/// Gets the number of values.
	BE_CORE_API uint4 ValueCount() const;
	/// Visits the given values.
	BE_CORE_API bool Visit(const PropertyProvider &provider, uint4 propertyID, const PropertyDesc &desc, void *values) LEAN_OVERRIDE;
};

/// Saves the given property provider to the given XML node.
BE_CORE_API void SaveProperties(const PropertyProvider &properties, rapidxml::xml_node<lean::utf8_t> &node, bool bPersistentOnly = true, bool bWithType = false);
/// Saves the given property provider to the given XML node.
BE_CORE_API void SaveProperties(const PropertyProvider &properties, uint4 propertyID, rapidxml::xml_node<lean::utf8_t> &node, bool bPersistentOnly = true, bool bWithType = false);

/// Load the given property provider from the given XML node.
BE_CORE_API void LoadProperties(PropertyProvider &properties, const rapidxml::xml_node<lean::utf8_t> &node);
/// Load the given property provider from the given XML node.
BE_CORE_API void LoadProperties(PropertyProvider &properties, uint4 propertyID, const rapidxml::xml_node<lean::utf8_t> &node);

} // namespace

#endif