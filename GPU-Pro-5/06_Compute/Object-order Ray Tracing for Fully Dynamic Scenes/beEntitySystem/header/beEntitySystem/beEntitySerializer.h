/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_ENTITYSERIALIZER
#define BE_ENTITYSYSTEM_ENTITYSERIALIZER

#include "beEntitySystem.h"
#include <beCore/beComponentSerializer.h>
#include "beEntities.h"

namespace beEntitySystem
{

/// Entity serializer.
class EntitySerializer : public beCore::ComponentSerializer<Entity>
{
public:
	/// Constructor.
	BE_ENTITYSYSTEM_API EntitySerializer();
	/// Destructor.
	BE_ENTITYSYSTEM_API ~EntitySerializer();

	/// Creates a serializable object from the given parameters.
	BE_ENTITYSYSTEM_API lean::scoped_ptr<Entity, lean::critical_ref> Create(
		const beCore::Parameters &creationParameters, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE;

	/// Loads an entity from the given xml node.
	BE_ENTITYSYSTEM_API virtual lean::scoped_ptr<Entity, lean::critical_ref> Load(const rapidxml::xml_node<lean::utf8_t> &node,
		beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const LEAN_OVERRIDE;
	/// Loads an entity from the given xml node.
	BE_ENTITYSYSTEM_API virtual void Load(Entity *entity, const rapidxml::xml_node<lean::utf8_t> &node,
		beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const LEAN_OVERRIDE;
	/// Saves the given entity object to the given XML node.
	BE_ENTITYSYSTEM_API virtual void Save(const Entity *entity, rapidxml::xml_node<lean::utf8_t> &node,
		beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue) const LEAN_OVERRIDE;
};

} // namespace

#endif