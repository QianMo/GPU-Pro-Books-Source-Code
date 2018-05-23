/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_SERIALIZATION_PARAMETERS
#define BE_ENTITYSYSTEM_SERIALIZATION_PARAMETERS

#include "beEntitySystem.h"
#include <beCore/beParameterSet.h>

namespace beEntitySystem
{

/// Gets the serialization parameter layout.
BE_ENTITYSYSTEM_API beCore::ParameterLayout& GetSerializationParameters();

// Prototypes
class World;
class Entity;

/// Scene parameter IDs.
struct EntitySystemParameterIDs
{
	uint4 World;
	uint4 Entity;
	uint4 NoOverwrite;

	/// Non-initializing constructor.
	EntitySystemParameterIDs() { }
	/// Constructor.
	EntitySystemParameterIDs(uint4 worldID, uint4 entityID, uint4 noOverwriteID)
			: World(worldID),
			Entity(entityID),
			NoOverwrite(noOverwriteID) { }
};

/// Scene parameters.
struct EntitySystemParameters
{
	World *World;
	Entity *Entity;

	/// Default constructor.
	EntitySystemParameters()
		: World(), Entity() { }
	/// Constructor.
	EntitySystemParameters(class World *world,
		class Entity *pEntity = nullptr)
			: World(world),
			Entity(pEntity) { }
};

/// Gets the serialization parameter IDs.
BE_ENTITYSYSTEM_API const EntitySystemParameterIDs& GetEntitySystemParameterIDs();

/// Sets the given entity system parameters in the given parameter set.
BE_ENTITYSYSTEM_API void SetEntitySystemParameters(beCore::ParameterSet &parameters, const EntitySystemParameters &entitySystemParameters);
/// Gets the given entity system parameters in the given parameter set.
BE_ENTITYSYSTEM_API EntitySystemParameters GetEntitySystemParameters(const beCore::ParameterSet &parameters);

/// Sets the given entity system parameters in the given parameter set.
BE_ENTITYSYSTEM_API void SetEntityParameter(beCore::ParameterSet &parameters, Entity *pEntity);
/// Gets the given entity system parameters in the given parameter set.
BE_ENTITYSYSTEM_API  Entity* GetEntityParameter(const beCore::ParameterSet &parameters);

/// Sets the given entity system parameters in the given parameter set.
BE_ENTITYSYSTEM_API void SetNoOverwriteParameter(beCore::ParameterSet &parameters, bool bNoOverwrite);
/// Gets the given entity system parameters in the given parameter set.
BE_ENTITYSYSTEM_API  bool GetNoOverwriteParameter(const beCore::ParameterSet &parameters);

} // namespace

#endif