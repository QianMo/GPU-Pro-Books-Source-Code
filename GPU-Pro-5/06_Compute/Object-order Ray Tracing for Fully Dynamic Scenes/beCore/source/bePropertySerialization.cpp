/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/bePropertySerialization.h"
#include "beCore/beTextSerializer.h"

#include <lean/functional/predicates.h>
#include <lean/xml/utility.h>
#include <lean/xml/numeric.h>
#include <sstream>

#include <lean/logging/log.h>

namespace beCore
{

// Visits the given values.
void PropertySerializer::Visit(const PropertyProvider &provider, uint4 propertyID, const PropertyDesc &desc, const void *values)
{
	rapidxml::xml_document<> &document = *LEAN_ASSERT_NOT_NULL(Parent->document());
	
	const utf8_t *value = nullptr;

	const TextSerializer *pSerializer = desc.TypeDesc->Text;

	if (pSerializer)
	{
		static const size_t StackBufferSize = 2048;

		size_t maxLength = pSerializer->GetMaxLength(desc.Count);

		// Use stack to speed up small allocations
		// NOTE: 0 means unpredictable
		if (maxLength != 0 && maxLength < StackBufferSize)
		{
			utf8_t buffer[StackBufferSize];
			
			// Serialize
			// NOTE: Required to be null-terminated -> xml
			utf8_t *bufferEnd = pSerializer->Write(buffer, desc.TypeDesc->Info.type, values, desc.Count);
			*bufferEnd++ = 0;

			size_t length = bufferEnd - buffer;
			LEAN_ASSERT(length < StackBufferSize);
			
			value = document.allocate_string(buffer, length);
		}
		// Take generic route, otherwise
		else
		{
			std::basic_ostringstream<utf8_t> stream;
			stream.imbue(std::locale::classic());

			// Serialize
			pSerializer->Write(stream, desc.TypeDesc->Info.type, values, desc.Count);

			value = document.allocate_string( stream.str().c_str() );
		}
	}
	else
		LEAN_LOG_ERROR("No serializer available for type \"" << desc.TypeDesc->Name.c_str());

	rapidxml::xml_node<utf8_t> &Node = *lean::allocate_node(document, "p", value);
	lean::append_attribute(document, Node, "n", Properties->GetPropertyName(propertyID));
	if (IncludeType)
	{
		lean::append_attribute(document, Node, "t", desc.TypeDesc->Name);
		lean::append_int_attribute(document, Node, "c", desc.Count);
	}
	Parent->append_node(&Node);
}

// Gets the property name.
utf8_ntr PropertyDeserializer::Name() const
{
	const utf8_t *chars;
	size_t count;

	if (Node->first_attribute())
	{
		chars = Node->first_attribute()->value();
		count = Node->first_attribute()->value_size();
	}
	else
	{
		chars = Node->name();
		count = Node->name_size();
	}

	return utf8_ntr(chars, chars + count);
}

// Gets the number of values.
utf8_ntr PropertyDeserializer::ValueType() const
{
	return lean::get_attribute(*Node, "t");
}

// Gets the number of values.
uint4 PropertyDeserializer::ValueCount() const
{
	return lean::get_int_attribute(*Node, "c", 1);
}

// Visits the given values.
bool PropertyDeserializer::Visit(const PropertyProvider &provider, uint4 propertyID, const PropertyDesc &desc, void *values)
{
	const TextSerializer *pSerializer = desc.TypeDesc->Text;

	if (pSerializer)
	{
		if (pSerializer->Read(Node->value(), Node->value() + Node->value_size(), desc.TypeDesc->Info.type, values, desc.Count))
			return true;
		// TODO: error logging?
	}
	else
		LEAN_LOG_ERROR("No serializer available for type \"" << desc.TypeDesc->Name.c_str());

	return false;
}

// Saves the given property provider to the given XML node.
void SaveProperties(const PropertyProvider &properties, rapidxml::xml_node<lean::utf8_t> &node, bool bPersistentOnly, bool bWithType)
{
	const uint4 propertyCount = properties.GetPropertyCount();
	uint4 propertyVisitFlags = (bPersistentOnly) ? PropertyVisitFlags::PersistentOnly : PropertyVisitFlags::None;

	if (propertyCount > 0)
	{
		rapidxml::xml_document<utf8_t> &document = *node.document();

		rapidxml::xml_node<utf8_t> &propertiesNode = *lean::allocate_node<utf8_t>(document, "properties");
		// ORDER: Append FIRST, otherwise parent document == nullptrs
		node.append_node(&propertiesNode);

		PropertySerializer serializer(properties, propertiesNode, bWithType);

		for (uint4 i = 0; i < propertyCount; ++i)
			properties.ReadProperty(i, serializer, propertyVisitFlags);
	}
}

// Saves the given property provider to the given XML node.
void SaveProperties(const PropertyProvider &properties, uint4 propertyID, rapidxml::xml_node<lean::utf8_t> &node, bool bPersistentOnly, bool bWithType)
{
	PropertySerializer serializer(properties, node);
	properties.ReadProperty(propertyID, serializer, (bPersistentOnly) ? PropertyVisitFlags::PersistentOnly : PropertyVisitFlags::None );
}

// Load the given property provider from the given XML node.
void LoadProperties(PropertyProvider &properties, const rapidxml::xml_node<lean::utf8_t> &node)
{
	const uint4 propertyCount = properties.GetPropertyCount();

	uint4 nextPropertyID = 0;

	for (const rapidxml::xml_node<lean::utf8_t> *propertiesNode = node.first_node("properties");
		propertiesNode; propertiesNode = propertiesNode->next_sibling("properties"))
		for (const rapidxml::xml_node<lean::utf8_t> *propertyNode = propertiesNode->first_node();
			propertyNode; propertyNode = propertyNode->next_sibling())
		{
			const utf8_t *nodeName = propertyNode->first_attribute() ? propertyNode->first_attribute()->value() : propertyNode->name();

			uint4 lowerPropertyID = nextPropertyID;
			uint4 upperPropertyID = nextPropertyID;

			for (uint4 i = 0; i < propertyCount; ++i)
			{
				// Perform bi-directional search: even == forward; odd == backward
				uint4 propertyID = (lean::is_odd(i) | (upperPropertyID == propertyCount)) & (lowerPropertyID != 0)
					? --lowerPropertyID
					: upperPropertyID++;

				if (nodeName == properties.GetPropertyName(propertyID))
				{
					PropertyDeserializer serializer(properties, *propertyNode);
					properties.WriteProperty(propertyID, serializer, PropertyVisitFlags::PersistentOnly);

					// Start next search with next property
					nextPropertyID = propertyID + 1;
					break;
				}
			}
		}
}

// Load the given property provider from the given XML node.
void LoadProperties(PropertyProvider &properties, uint4 propertyID, const rapidxml::xml_node<lean::utf8_t> &node)
{
	utf8_ntr propertyName = properties.GetPropertyName(propertyID);

	for (const rapidxml::xml_node<lean::utf8_t> *propertiesNode = node.first_node("properties");
		propertiesNode; propertiesNode = propertiesNode->next_sibling("properties"))
		for (const rapidxml::xml_node<lean::utf8_t> *propertyNode = propertiesNode->first_node();
			propertyNode; propertyNode = propertyNode->next_sibling())
		{
			const utf8_t *nodeName = propertyNode->first_attribute() ? propertyNode->first_attribute()->value() : propertyNode->name();

			if (nodeName == propertyName)
			{
				PropertyDeserializer serializer(properties, *propertyNode);
				properties.WriteProperty(propertyID, serializer, PropertyVisitFlags::PersistentOnly);
			}
		}
}

} // namespace
