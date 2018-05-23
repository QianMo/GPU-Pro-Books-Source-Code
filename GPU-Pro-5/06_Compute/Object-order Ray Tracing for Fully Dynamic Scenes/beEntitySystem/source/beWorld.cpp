/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beWorld.h"
#include "beEntitySystem/beEntities.h"

#include "beEntitySystem/beWorldControllers.h"

#include "beEntitySystem/beSimulation.h"

#include "beEntitySystem/beEntitySerialization.h"
#include "beEntitySystem/beSerializationParameters.h"
#include "beEntitySystem/beSerializationTasks.h"

#include <lean/functional/algorithm.h>

#include <lean/xml/xml_file.h>
#include <lean/xml/utility.h>
#include <lean/xml/numeric.h>

#include <lean/logging/errors.h>

namespace beEntitySystem
{

// Creates an empty world.
World::World(const utf8_ntri &name, lean::move_ptr<WorldControllers> pTmpControllers, const WorldDesc &desc)
	: m_name(name.to<utf8_string>()),
	m_desc(desc),
	m_entities( CreateEntities(&m_persistentIDs) ),
	m_controllers( (pTmpControllers.peek()) ? pTmpControllers.transfer() : new WorldControllers() )
{
}

// Loads the world from the given file.
World::World(const utf8_ntri &name, const utf8_ntri &file, beCore::ParameterSet &parameters, lean::move_ptr<WorldControllers> pTmpControllers, const WorldDesc &desc)
	: m_name(name.to<utf8_string>()),
	m_desc(desc),
	m_entities( CreateEntities(&m_persistentIDs) ),
	m_controllers( (pTmpControllers.peek()) ? pTmpControllers.transfer() : new WorldControllers() )
{
	lean::xml_file<lean::utf8_t> xml(file);
	rapidxml::xml_node<lean::utf8_t> *root = xml.document().first_node("world");

	if (root)
		LoadWorld(*root, parameters);
	else
		LEAN_THROW_ERROR_CTX("No world node found", file.c_str());
}

// Loads the world from the given XML node.
World::World(const utf8_ntri &name, const rapidxml::xml_node<lean::utf8_t> &node, beCore::ParameterSet &parameters, lean::move_ptr<WorldControllers> pTmpControllers, const WorldDesc &desc)
	: m_name(name.to<utf8_string>()),
	m_desc(desc),
	m_entities( CreateEntities(&m_persistentIDs) ),
	m_controllers( (pTmpControllers.peek()) ? pTmpControllers.transfer() : new WorldControllers() )
{
	LoadWorld(node, parameters);
}

// Destructor.
World::~World()
{
}

// Checks for and commits all changes.
void World::Commit()
{
	m_entities->Commit();
	bees::Commit(m_controllers->GetControllers());
}

// Saves the world to the given file.
void World::Serialize(const lean::utf8_ntri &file) const
{
	lean::xml_file<lean::utf8_t> xml;

	rapidxml::xml_node<lean::utf8_t> &root = *lean::allocate_node<utf8_t>(xml.document(), "world");
	// ORDER: Append FIRST, otherwise parent document == nullptr
	xml.document().append_node(&root);
	
	Serialize(root);

	xml.save(file);
}

// Saves the world to the given XML node.
void World::Serialize(rapidxml::xml_node<lean::utf8_t> &node) const
{
	SaveWorld(node);
}

// Saves the world to the given xml node.
void World::SaveWorld(rapidxml::xml_node<utf8_t> &worldNode) const
{
	rapidxml::xml_document<utf8_t> &document = *worldNode.document();

	lean::append_attribute<utf8_t>(document, worldNode, "name", m_name);
	
	// NOTE: Never re-use persistent IDs again
	lean::append_int_attribute<utf8_t>(document, worldNode, "nextPersistentID", m_persistentIDs.GetNextID());

	beCore::ParameterSet parameters(&GetSerializationParameters());
	
	// Execute generic save tasks first
	GetResourceSaveTasks().Save(worldNode, parameters);
	GetWorldSaveTasks().Save(worldNode, parameters);
	
	beCore::SaveJobs saveJobs;

	Entities::ConstRange entities = m_entities->GetEntities();
	SaveEntities(&entities[0], Size4(entities), worldNode, &parameters, &saveJobs);

	// Execute any additionally scheduled save jobs
	saveJobs.Save(worldNode, parameters);
}

// Loads the world from the given xml node.
void World::LoadWorld(const rapidxml::xml_node<lean::utf8_t> &worldNode, beCore::ParameterSet &parameters)
{
	lean::get_attribute<utf8_t>(worldNode, "name", m_name);

	// NOTE: Never re-use persistent IDs again
	m_persistentIDs.SkipIDs( lean::get_int_attribute<utf8_t>(worldNode, "nextPersistentID", m_persistentIDs.GetNextID()) );

	// NOTE: Caller has no way of setting this right!
	SetEntitySystemParameters(
			parameters,
			EntitySystemParameters(this)
		);

	// Execute generic load tasks first
	GetResourceLoadTasks().Load(worldNode, parameters);
	GetWorldLoadTasks().Load(worldNode, parameters);

	beCore::LoadJobs loadJobs;
	LoadEntities(m_entities.get(), worldNode, parameters, &loadJobs);

	// Execute any additionally scheduled load jobs
	loadJobs.Load(worldNode, parameters);
}

// Sets the name.
void World::SetName(const utf8_ntri &name)
{
	m_name.assign(name.begin(), name.end());
}

// Attaches the given collection of entitites.
void Attach(World *world, Simulation *simulation)
{
	Attach(world->Entities(), simulation);
	Attach(world->Controllers().GetControllers(), simulation);
}

// Detaches the given collection of entitites.
void Detach(World *world, Simulation *simulation)
{
	Detach(world->Controllers().GetControllers(), simulation);
	Detach(world->Entities(), simulation);
}

// Attaches the given collection of simulation controllers.
void Attach(Entities *entities, Simulation *simulation)
{
	simulation->AddSynchronized(entities, SynchronizedFlags::Flush);
}

// Detaches the given collection of simulation controllers.
void Detach(Entities *entities, Simulation *simulation)
{
	simulation->RemoveSynchronized(entities, SynchronizedFlags::All);
}

} // namespace
