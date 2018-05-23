/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_WORLD
#define BE_ENTITYSYSTEM_WORLD

#include "beEntitySystem.h"
#include <lean/tags/noncopyable.h>
#include <beCore/beShared.h>
#include <lean/smart/resource_ptr.h>
#include <vector>
#include <beCore/bePersistentIDs.h>
#include "beWorldControllers.h"
#include <lean/rapidxml/rapidxml.hpp>
#include <lean/smart/scoped_ptr.h>

// Prototypes
namespace beCore
{
	class ParameterSet;
}

namespace beEntitySystem
{

// Prototypes
class Entity;
class Entities;
class Assets;

/// World description.
struct WorldDesc
{
	// TODO: Remove
	int4 CellSize;		///< Size of one world cell.

	/// Constructor.
	WorldDesc(int4 cellSize = 10000)
		: CellSize(cellSize) { }
};

/// World class.
class World : public lean::noncopyable_chain<beCore::Resource>
{
private:
	utf8_string m_name;
	WorldDesc m_desc;
	
	beCore::PersistentIDs m_persistentIDs;

	lean::scoped_ptr<WorldControllers> m_controllers;
	lean::scoped_ptr<Entities> m_entities;
//	lean::scoped_ptr<Assets> m_assets;

	/// Saves the world to the given xml node.
	void SaveWorld(rapidxml::xml_node<lean::utf8_t> &node) const;
	/// Loads the world from the given xml node.
	void LoadWorld(const rapidxml::xml_node<lean::utf8_t> &node, beCore::ParameterSet &parameters);

public:
	/// Creates an empty world.
	BE_ENTITYSYSTEM_API explicit World(const utf8_ntri &name, lean::move_ptr<WorldControllers> pControllers = nullptr, const WorldDesc &desc = WorldDesc());
	/// Loads the world from the given file.
	BE_ENTITYSYSTEM_API explicit World(const utf8_ntri &name, const utf8_ntri &file, beCore::ParameterSet &parameters, lean::move_ptr<WorldControllers> pControllers = nullptr, const WorldDesc &desc = WorldDesc());
	/// Loads the world from the given XML node.
	BE_ENTITYSYSTEM_API explicit World(const utf8_ntri &name, const rapidxml::xml_node<lean::utf8_t> &node, beCore::ParameterSet &parameters, lean::move_ptr<WorldControllers> pControllers = nullptr, const WorldDesc &desc = WorldDesc());
	/// Destructor.
	BE_ENTITYSYSTEM_API virtual ~World();

	/// Gets the entity manager.
	BE_ENTITYSYSTEM_API class Entities* Entities() { return m_entities.get(); }
	/// Gets the entity manager.
	BE_ENTITYSYSTEM_API const class Entities* Entities() const { return m_entities.get(); }

	/// Gets the controller manager.
	BE_ENTITYSYSTEM_API WorldControllers& Controllers() { return *m_controllers.get(); }
	/// Gets the controller manager.
	BE_ENTITYSYSTEM_API const WorldControllers& Controllers() const { return *m_controllers.get(); }

	/// Checks for and commits all changes.
	BE_ENTITYSYSTEM_API void Commit();

	/// Gets the asset manager.
//	BE_ENTITYSYSTEM_API beEntitySystem::Assets& Assets() { return *m_assets; }
	/// Gets the asset manager.
//	BE_ENTITYSYSTEM_API const beEntitySystem::Assets& Assets() const{ return *m_assets; }

	/// Saves the world to the given file.
	BE_ENTITYSYSTEM_API void Serialize(const lean::utf8_ntri &file) const;
	/// Saves the world to the given XML node.
	BE_ENTITYSYSTEM_API void Serialize(rapidxml::xml_node<lean::utf8_t> &node) const;

	/// Gets the world's persistent IDs.
	LEAN_INLINE beCore::PersistentIDs& PersistentIDs() { return m_persistentIDs; }
	/// Gets the world's persistent IDs.
	LEAN_INLINE const beCore::PersistentIDs& PersistentIDs() const { return m_persistentIDs; }

	/// Gets the world's cell size.
	LEAN_INLINE int4 GetCellSize() { return m_desc.CellSize; }

	/// Sets the name.
	BE_ENTITYSYSTEM_API void SetName(const utf8_ntri &name);
	/// Gets the name.
	LEAN_INLINE const utf8_string& GetName() const { return m_name; }
};

/// Attaches the given collection of entitites.
BE_ENTITYSYSTEM_API void Attach(World *world, Simulation *simulation);
/// Detaches the given collection of entitites.
BE_ENTITYSYSTEM_API void Detach(World *world, Simulation *simulation);

/// Attaches the given collection of entitites.
BE_ENTITYSYSTEM_API void Attach(Entities *entities, Simulation *simulation);
/// Detaches the given collection of entitites.
BE_ENTITYSYSTEM_API void Detach(Entities *entities, Simulation *simulation);

} // namespace

#endif