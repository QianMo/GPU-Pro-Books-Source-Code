/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_ENTITYSERIALIZATION
#define BE_ENTITYSYSTEM_ENTITYSERIALIZATION

#include "beEntitySystem.h"
#include <lean/rapidxml/rapidxml.hpp>

namespace beCore
{
	class ParameterSet;
	class SaveJobs;
	class LoadJobs;
}

namespace beEntitySystem
{

class Entity;
class Entities;

class LEAN_INTERFACE EntityInserter
{
	LEAN_INTERFACE_BEHAVIOR(EntityInserter)

public:
	virtual void Reserve(uint4 count) = 0;
	virtual void Add(Entity *entity) = 0;
};

/// Saves the given number of entities to the given xml node.
BE_ENTITYSYSTEM_API void SaveEntities(const Entity *const *entities, uint4 entityCount, rapidxml::xml_node<utf8_t> &parentNode,
	beCore::ParameterSet *pParameters = nullptr, beCore::SaveJobs *pQueue = nullptr);
/// Loads all entities from the given xml node.
BE_ENTITYSYSTEM_API void LoadEntities(Entities* entities, const rapidxml::xml_node<lean::utf8_t> &parentNode,
	beCore::ParameterSet &parameters, beCore::LoadJobs *pQueue = nullptr, EntityInserter *pInserter = nullptr);

} // namespace

#endif