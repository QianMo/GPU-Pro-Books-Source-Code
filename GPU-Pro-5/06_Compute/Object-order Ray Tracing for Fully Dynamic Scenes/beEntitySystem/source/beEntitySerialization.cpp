/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beEntitySerialization.h"
#include "beEntitySystem/beEntities.h"
#include "beEntitySystem/beEntitySerializer.h"

#include "beEntitySystem/beSerializationParameters.h"
#include "beEntitySystem/beSerialization.h"
#include "beEntitySystem/beSerializationTasks.h"

#include <lean/xml/utility.h>
#include <lean/smart/scoped_ptr.h>
#include <lean/logging/errors.h>

namespace beEntitySystem
{

// Saves the given number of entities to the given xml node.
void SaveEntities(const Entity *const *entities, uint4 entityCount, rapidxml::xml_node<utf8_t> &parentNode, 
				  beCore::ParameterSet *pParameters, beCore::SaveJobs *pQueue)
{
	rapidxml::xml_document<utf8_t> &document = *parentNode.document();

	rapidxml::xml_node<utf8_t> &entitiesNode = *lean::allocate_node<utf8_t>(document, "entities");
	// ORDER: Append FIRST, otherwise parent document == nullptrs
	parentNode.append_node(&entitiesNode);

	lean::scoped_ptr<beCore::ParameterSet> pPrivateParameters;

	if (!pParameters)
	{
		pPrivateParameters = new beCore::ParameterSet(&GetSerializationParameters());
		pParameters = pPrivateParameters.get();
	}

	lean::scoped_ptr<beCore::SaveJobs> pPrivateSaveJobs;

	if (!pQueue)
	{
		pPrivateSaveJobs = new beCore::SaveJobs();
		pQueue = pPrivateSaveJobs.get();
	}

	const EntitySerialization &entitySerialization = GetEntitySerialization();

	for (const Entity *const *itEntity = entities, *const *itEntityEnd = entities + entityCount; itEntity < itEntityEnd; ++itEntity)
	{
		const Entity *entity = *itEntity;

		if (entity->IsAttached() && entity->IsSerialized())
		{
			rapidxml::xml_node<utf8_t> &entityNode = *lean::allocate_node<utf8_t>(document, "e");
			// ORDER: Append FIRST, otherwise parent document == nullptr
			entitiesNode.append_node(&entityNode);

			entitySerialization.Save(entity, entityNode, *pParameters, *pQueue);
		}
	}

	// Execute any additionally scheduled save jobs
	if (pPrivateSaveJobs)
		pPrivateSaveJobs->Save(parentNode, *pParameters);
}

// Loads all entities from the given xml node.
void LoadEntities(Entities *entities, const rapidxml::xml_node<lean::utf8_t> &parentNode,
				  beCore::ParameterSet &parameters, beCore::LoadJobs *pQueue, EntityInserter *pInserter)
{
	lean::scoped_ptr<beCore::LoadJobs> pPrivateLoadJobs;

	if (!pQueue)
	{
		pPrivateLoadJobs = new beCore::LoadJobs();
		pQueue = pPrivateLoadJobs.get();
	}

	const EntitySerialization &entitySerialization = GetEntitySerialization();

	for (const rapidxml::xml_node<utf8_t> *pEntitiesNode = parentNode.first_node("entities");
		pEntitiesNode; pEntitiesNode = pEntitiesNode->next_sibling("entities"))
	{
		uint4 predictedCount = lean::node_count(*pEntitiesNode);
		entities->Reserve(predictedCount);
		if (pInserter)
			pInserter->Reserve(predictedCount);

		for (const rapidxml::xml_node<utf8_t> *pEntityNode = pEntitiesNode->first_node();
			pEntityNode; pEntityNode = pEntityNode->next_sibling())
		{
			lean::scoped_ptr<Entity> pEntity = entitySerialization.Load(*pEntityNode, parameters, *pQueue);

			if (pEntity)
			{
				if (pInserter)
					pInserter->Add(pEntity);

				// Success
				pEntity.detach();
			}
			else
				LEAN_LOG_ERROR_CTX("LoadEntities()", EntitySerializer::GetName(*pEntityNode));
		}
	}

	// Execute any additionally scheduled load jobs
	if (pPrivateLoadJobs)
		pPrivateLoadJobs->Load(parentNode, parameters);
}

} // namespace
