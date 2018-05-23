/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beEntitySerializer.h"
#include "beEntitySystem/beEntities.h"
#include "beEntitySystem/beEntityController.h"

#include "beEntitySystem/beWorld.h"

#include "beEntitySystem/beControllerSerializer.h"

#include "beEntitySystem/beSerializationParameters.h"
#include <beCore/bePropertySerialization.h>
#include <beCore/beParameters.h>

#include "beEntitySystem/beSerialization.h"

#include <lean/xml/utility.h>
#include <lean/logging/errors.h>

namespace beEntitySystem
{

// Constructor.
EntitySerializer::EntitySerializer()
	: ComponentSerializer<Entity>("Entity")
{
}

// Destructor.
EntitySerializer::~EntitySerializer()
{
}

namespace
{

Entities* GetEntities(const beCore::ParameterSet &parameters)
{
	EntitySystemParameters entityParameters = GetEntitySystemParameters(parameters);
	return LEAN_THROW_NULL(entityParameters.World)->Entities();
}


// Loads all controllers from the given xml node.
void LoadControllers(Entity *entity, const rapidxml::xml_node<lean::utf8_t> &node, 
	beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue)
{
	const EntityControllerSerialization &controllerSerialization = GetEntityControllerSerialization();

	for (const rapidxml::xml_node<utf8_t> *pControllersNode = node.first_node("controllers");
		pControllersNode; pControllersNode = pControllersNode->next_sibling("controllers"))
		for (const rapidxml::xml_node<utf8_t> *pControllerNode = pControllersNode->first_node();
			pControllerNode; pControllerNode = pControllerNode->next_sibling())
		{
			lean::scoped_ptr<EntityController> pController = controllerSerialization.Load(*pControllerNode, parameters, queue);

			if (pController)
				entity->AddController(pController.move_ptr());
			else
				LEAN_LOG_ERROR_CTX("ControllerSerialization::Load()", beCore::ComponentSerializer<EntityController>::GetName(*pControllerNode));
		}
}

// Saves all controllers of the given serializable object to the given XML node.
void SaveControllers(const Entity *entity, rapidxml::xml_node<lean::utf8_t> &node,
	beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue)
{
	Entity::Controllers controllers = entity->GetControllers();

	if (Size(controllers))
	{
		rapidxml::xml_document<utf8_t> &document = *node.document();

		rapidxml::xml_node<utf8_t> &controllersNode = *lean::allocate_node<utf8_t>(document, "controllers");
		// ORDER: Append FIRST, otherwise parent document == nullptrs
		node.append_node(&controllersNode);

		const EntityControllerSerialization &controllerSerialization = GetEntityControllerSerialization();

		for (; controllers.Begin < controllers.End; ++controllers.Begin)
		{
			rapidxml::xml_node<utf8_t> &controllerNode = *lean::allocate_node<utf8_t>(document, "c");
			// ORDER: Append FIRST, otherwise document == nullptr
			controllersNode.append_node(&controllerNode);

			controllerSerialization.Save(*controllers.Begin, controllerNode, parameters, queue);
		}
	}
}

} // namespace

// Creates a serializable object from the given parameters.
lean::scoped_ptr<Entity, lean::critical_ref> EntitySerializer::Create(const beCore::Parameters &creationParameters, const beCore::ParameterSet &parameters) const
{
	lean::scoped_ptr<Entity> entity( GetEntities(parameters)->AddEntity() );
	entity->SetName( creationParameters.GetValueDefault<beCore::Exchange::utf8_string>("Name") );

	return entity.transfer();
}

// Loads an entity from the given xml node.
lean::scoped_ptr<Entity, lean::critical_ref> EntitySerializer::Load(const rapidxml::xml_node<lean::utf8_t> &node,
	beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const
{
	lean::scoped_ptr<Entity> entity( GetEntities(parameters)->AddEntity() );
	entity->SetName( GetName(node) );

	Load(entity.get(), node, parameters, queue);
	
	return entity.transfer();
}

// Loads an entity from the given xml node.
void EntitySerializer::Load(Entity *entity, const rapidxml::xml_node<lean::utf8_t> &node,
	beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const
{
	ComponentSerializer<Entity>::Load(entity, node, parameters, queue);
	entity->SetPersistentID( EntitySerializer::GetID(node) );

	// Properties
	LoadProperties(*entity, node);

	// Controllers
	SetEntityParameter(parameters, entity);
	LoadControllers(entity, node, parameters, queue);
}

// Saves the given entity object to the given XML node.
void EntitySerializer::Save(const Entity *entity, rapidxml::xml_node<lean::utf8_t> &node,
	beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue) const
{
	ComponentSerializer<Entity>::Save(entity, node, parameters, queue);
	SetName(entity->GetName(), node);
	SetID(entity->GetPersistentID(), node);

	// Properties
	SaveProperties(*entity, node);
	
	// Controllers
	SaveControllers(entity, node, parameters, queue);
}

namespace
{
	
const beCore::ComponentSerializationPlugin<EntitySerializer, EntitySerialization, GetEntitySerialization> ESP;

}

} // namespace
