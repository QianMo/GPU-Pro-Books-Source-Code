/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beComponentSerialization.h"
#include "beCore/beComponentSerializer.h"
#include "beCore/beComponent.h"
#include <lean/xml/utility.h>

namespace beCore
{

// Constructor.
GenericComponentSerialization::GenericComponentSerialization()
{
}

// Destructor.
GenericComponentSerialization::~GenericComponentSerialization()
{
}

// Adds the given serializer to this serialization manager.
void GenericComponentSerialization::AddSerializer(const Serializer *serializer)
{
	LEAN_ASSERT(serializer);

	m_serializers[serializer->GetType()] = serializer;
}

// Removes the given serializer from this serialization manager.
bool GenericComponentSerialization::RemoveSerializer(const Serializer *serializer)
{
	return (m_serializers.erase(serializer->GetType()) != 0);
}

// Gets the number of serializers.
uint4 GenericComponentSerialization::GetSerializerCount() const
{
	return static_cast<uint4>(m_serializers.size());
}

// Gets all serializers.
void GenericComponentSerialization::GetSerializers(const Serializer **serializers) const
{
	for (serializer_map::const_iterator it = m_serializers.begin(); it != m_serializers.end(); ++it)
		*serializers++ = it->second;
}

// Gets an entity serializer for the given entity type, if available, returns nullptr otherwise.
const GenericComponentSerialization::Serializer* GenericComponentSerialization::GetSerializer(const utf8_ntri &type) const
{
	serializer_map::const_iterator itSerializer = m_serializers.find(type.to<utf8_string>());

	return (itSerializer != m_serializers.end())
		? itSerializer->second
		: nullptr;
}

// Loads an entity from the given xml node.
GenericComponentSerialization::Serializable* GenericComponentSerialization::Load(const rapidxml::xml_node<lean::utf8_t> &node,
	beCore::ParameterSet &parameters, SerializationQueue<LoadJob> &queue) const
{
	const Serializer *pSerializer = GetSerializer( Serializer::GetType(node) );
	
	return (pSerializer)
		? pSerializer->LoadComponent(node, parameters, queue)
		: nullptr;
}

// Saves the given entity to the given XML node.
bool GenericComponentSerialization::Save(const Serializable *serializable, rapidxml::xml_node<lean::utf8_t> &node,
	beCore::ParameterSet &parameters, SerializationQueue<SaveJob> &queue) const
{
	const Serializer *pSerializer = GetSerializer( serializable->GetType()->Name );
	
	if (pSerializer)
		pSerializer->SaveComponent(serializable, node, parameters, queue);

	return (pSerializer != nullptr);
}

} // namespace
