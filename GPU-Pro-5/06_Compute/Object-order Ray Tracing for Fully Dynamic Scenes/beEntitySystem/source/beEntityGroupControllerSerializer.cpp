/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beGenericControllerSerializer.h"

#include "beEntitySystem/beEntityGroupController.h"

#include <beEntitySystem/beSerializationParameters.h>
#include <beEntitySystem/beSerializationTasks.h>
#include <beEntitySystem/beSerialization.h>

#include <beEntitySystem/beWorld.h>
#include <beEntitySystem/beEntities.h>

#include <lean/xml/numeric.h>
#include <lean/logging/errors.h>

namespace beEntitySystem
{

namespace
{

/// Assigns grouped entities to their groups.
struct EntityGroupLinker : public beCore::LoadJob
{
public:
	struct Group
	{
		EntityGroupController *controller;
		const rapidxml::xml_node<utf8_t> *node;

		Group(EntityGroupController *controller, const rapidxml::xml_node<utf8_t> *node)
			: controller(controller),
			node(node) { }
	};

	typedef std::vector<Group> groups_t;
	groups_t groups;

	/// Loads anything, e.g. to the given XML root node.
	void Load(const rapidxml::xml_node<lean::utf8_t> &root, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const
	{
		World &world = *LEAN_ASSERT_NOT_NULL( GetEntitySystemParameters(parameters).World );
		bec::PersistentIDs &persistentIDs = world.PersistentIDs();

		for (groups_t::const_iterator it = groups.begin(), itEnd = groups.end(); it < itEnd; ++it)
		{
			for (const rapidxml::xml_node<utf8_t> *entitiesNode = it->node->first_node("entities");
				entitiesNode; entitiesNode = entitiesNode->next_sibling("entities"))
			{
				it->controller->ReserveNextEntities( lean::node_count(*entitiesNode) );

				for (const rapidxml::xml_node<utf8_t> *entityNode = entitiesNode->first_node();
					entityNode; entityNode = entityNode->next_sibling())
				{
					uint8 id = lean::get_int_attribute(*entityNode, "id", bec::PersistentIDs::InvalidID);

					if (Entity *entity = persistentIDs.GetReference<Entity>(id))
						it->controller->AddEntity(entity);
					else
						// TODO: Context
						LEAN_LOG_ERROR_MSG("Cannot identify grouped entity, will be lost");
				}
			}
		}
	}
};

struct InlineSerializationToken;

/// Adds the given entity group for linking.
void LinkEntityGroup(EntityGroupController *controller, const rapidxml::xml_node<lean::utf8_t> &node,
					 beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue)
{
	// Schedule group for linking
	GetOrMakeLoadJob<EntityGroupLinker, InlineSerializationToken>(
			parameters, "beEntitySystem.EntityGroupLinker", queue
		).groups.push_back( EntityGroupLinker::Group(controller, &node) );
}

/// Saves links to the grouped entities.
void SaveEntityLinks(rapidxml::xml_node<lean::utf8_t> &node, const Entity *const *entities, uint4 count)
{
	rapidxml::xml_document<utf8_t> &document = *node.document();

	if (count > 0)
	{
		rapidxml::xml_node<utf8_t> &entitiesNode = *lean::allocate_node<utf8_t>(document, "entities");
		// ORDER: Append FIRST, otherwise parent document == nullptrs
		node.append_node(&entitiesNode);

		for (const Entity *const *it = entities, *const *itEnd = entities + count; it < itEnd; ++it)
		{
			rapidxml::xml_node<utf8_t> &entityNode = *lean::allocate_node<utf8_t>(document, "e");
			lean::append_int_attribute( document, entityNode, "id", (*it)->GetPersistentID() );
			// ORDER: Append FIRST, otherwise parent document == nullptr
			entitiesNode.append_node(&entityNode);
		}
	}
}

} // namespace

class EntityGroupControllerSerializer : public GenericControllerSerializer<EntityGroupController>
{
	// Loads a mesh controller from the given xml node.
	lean::scoped_ptr<Controller, lean::critical_ref> Load(const rapidxml::xml_node<lean::utf8_t> &node,
		beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const
	{
		lean::scoped_ptr<Controller> controller = GenericControllerSerializer::Load(node, parameters, queue);
		LinkEntityGroup(static_cast<EntityGroupController*>(controller.get()), node, parameters, queue);
		return controller.transfer();
	}

	// Saves the given mesh controller to the given XML node.
	void Save(const Controller *serializable, rapidxml::xml_node<lean::utf8_t> &node,
		beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue) const
	{
		ControllerSerializer::Save(serializable, node, parameters, queue);
		EntityGroupController::ConstEntityRange entities = static_cast<const EntityGroupController*>(serializable)->GetEntities();
		SaveEntityLinks(node, entities.Begin, Size4(entities));
	}
};

const bees::EntityControllerSerializationPlugin< bees::EntityGroupControllerSerializer > ControllerSerializerPlugin;

} // namespace
