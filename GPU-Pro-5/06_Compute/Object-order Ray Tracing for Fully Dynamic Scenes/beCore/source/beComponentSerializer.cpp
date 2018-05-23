/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beComponentSerializer.h"
#include "beCore/beComponent.h"
#include <lean/xml/utility.h>
#include <lean/xml/numeric.h>

namespace beCore
{

// Constructor.
GenericComponentSerializer::GenericComponentSerializer(const utf8_ntri &tye)
	: m_type(tye.to<utf8_string>())
{
}

// Destructor.
GenericComponentSerializer::~GenericComponentSerializer()
{
}

// Gets the name of the serializable object stored in the given xml node.
utf8_ntr GenericComponentSerializer::GetName(const rapidxml::xml_node<lean::utf8_t> &node)
{
	return lean::get_attribute(node, "name");
}

/// Sets the name of the serializable object stored in the given xml node.
void GenericComponentSerializer::SetName(const utf8_ntri &name, rapidxml::xml_node<lean::utf8_t> &node)
{
	lean::append_attribute(*node.document(), node, "name", name);
}

// Gets the type of the serializable object stored in the given xml node.
utf8_ntr GenericComponentSerializer::GetType(const rapidxml::xml_node<lean::utf8_t> &node)
{
	return lean::get_attribute(node, "type");
}

// Sets the type of the serializable object stored in the given xml node.
void GenericComponentSerializer::SetType(const utf8_ntri &type, rapidxml::xml_node<lean::utf8_t> &node)
{
	lean::append_attribute(*node.document(), node, "type", type);
}

// Gets the ID of the serializable object stored in the given xml node.
uint8 GenericComponentSerializer::GetID(const rapidxml::xml_node<lean::utf8_t> &node)
{
	return lean::get_int_attribute(node, "id", static_cast<uint8>(-1));
}

// Sets the ID of the serializable object stored in the given xml node.
void GenericComponentSerializer::SetID(uint8 id, rapidxml::xml_node<lean::utf8_t> &node)
{
	lean::append_int_attribute(*node.document(), node, "id", id);
}

// Gets a list of creation parameters.
ComponentParameters GenericComponentSerializer::GetCreationParameters() const
{
	return ComponentParameters(nullptr, nullptr);
}

// Loads a serializable object from the given xml node.
void GenericComponentSerializer::LoadComponent(Serializable *serializable, const rapidxml::xml_node<lean::utf8_t> &node, 
	beCore::ParameterSet &parameters, SerializationQueue<LoadJob> &queue) const
{
}

// Saves the given serializable object to the given XML node.
void GenericComponentSerializer::SaveComponent(const Serializable *serializable, rapidxml::xml_node<lean::utf8_t> &node,
	beCore::ParameterSet &parameters, SerializationQueue<SaveJob> &queue) const
{
	SetType(serializable->GetType()->Name, node);
}

} // namespace
