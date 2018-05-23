/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beEntityGroup.h"
#include "beEntitySystem/beAsset.h"
#include "beEntitySystem/beEntities.h"

#include "beEntitySystem/beSerialization.h"
#include "beEntitySystem/beSerializationTasks.h"
#include "beEntitySystem/beEntitySerialization.h"

#include <lean/xml/xml_file.h>
#include <lean/xml/utility.h>
#include <lean/xml/numeric.h>

#include <lean/logging/errors.h>

namespace beEntitySystem
{

// Saves resources to the given xml node.
void SaveResources(rapidxml::xml_node<lean::utf8_t> &parentNode, beCore::ParameterSet &parameters)
{
	GetResourceSaveTasks().Save(parentNode, parameters);
}

// Saves entities and resources to the given xml node.
void SaveAsset(const EntityGroup &group, rapidxml::xml_node<lean::utf8_t> &parentNode, beCore::ParameterSet *pParameters)
{
	rapidxml::xml_document<utf8_t> &document = *parentNode.document();

	// Save resources first
	if (pParameters)
		SaveResources(parentNode, *pParameters);

	EntityGroup::ConstEntityRange entities = group.GetEntities();
	SaveEntities(entities.begin(), static_cast<uint4>(entities.size()), parentNode, pParameters);
}

// Saves entities and resources to the given xml node.
void SaveAsset(const EntityGroup &group, const utf8_ntri &file, beCore::ParameterSet *pParameters)
{
	lean::xml_file<lean::utf8_t> xml;

	rapidxml::xml_node<lean::utf8_t> &root = *lean::allocate_node<utf8_t>(xml.document(), "world");
	// ORDER: Append FIRST, otherwise parent document == nullptr
	xml.document().append_node(&root);

	SaveAsset(group, root, pParameters);

	xml.save(file);
}

// Loads resources from the given xml node.
void LoadResources(const rapidxml::xml_node<lean::utf8_t> &parentNode, beCore::ParameterSet &parameters)
{
	GetResourceLoadTasks().Load(parentNode, parameters);
}

// Loads entities and resources from the given xml node.
void LoadAsset(Entities *entities, EntityGroup &group, const rapidxml::xml_node<lean::utf8_t> &parentNode, beCore::ParameterSet &parameters)
{
	// Load resources first
	LoadResources(parentNode, parameters);

	struct GroupInserter : public EntityInserter
	{
		EntityGroup *group;

		GroupInserter(EntityGroup &group)
			: group(&group) { }
		
		void Reserve(uint4 count) { }

		void Add(Entity *pEntity)
		{
			group->AddEntity(pEntity);
		}
	} inserter(group);

	LoadEntities(entities, parentNode, parameters, nullptr, &inserter);
}

// Loads entities and resources from the given xml node.
void LoadAsset(Entities *entities, EntityGroup &group, const utf8_ntri &file, beCore::ParameterSet &parameters)
{
	lean::xml_file<lean::utf8_t> xml(file);
	rapidxml::xml_node<lean::utf8_t> *root = xml.document().first_node();

	if (root)
		LoadAsset(entities, group, *root, parameters);
	else
		LEAN_THROW_ERROR_CTX("No asset root found", file.c_str());
}

} // namespace
