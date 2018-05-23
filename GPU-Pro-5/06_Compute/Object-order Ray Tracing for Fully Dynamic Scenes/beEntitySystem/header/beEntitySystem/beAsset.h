/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_ASSET
#define BE_ENTITYSYSTEM_ASSET

#include "beEntitySystem.h"
#include <lean/rapidxml/rapidxml.hpp>

// Prototypes
namespace beCore
{
	class ParameterSet;
}

namespace beEntitySystem
{

class Entities;
class EntityGroup;

/// Saves resources to the given xml node.
BE_ENTITYSYSTEM_API void SaveResources(rapidxml::xml_node<lean::utf8_t> &parentNode, beCore::ParameterSet &parameters);
/// Saves entities and resources to the given xml node.
BE_ENTITYSYSTEM_API void SaveAsset(const EntityGroup &group, rapidxml::xml_node<lean::utf8_t> &parentNode, beCore::ParameterSet *pParameters = nullptr);

/// Saves entities and resources to the given xml node.
BE_ENTITYSYSTEM_API void SaveAsset(const EntityGroup &group, const utf8_ntri &file, beCore::ParameterSet *pParameters = nullptr);

/// Loads resources from the given xml node.
BE_ENTITYSYSTEM_API void LoadResources(const rapidxml::xml_node<lean::utf8_t> &parentNode, beCore::ParameterSet &parameters);
/// Loads entities and resources from the given xml node.
BE_ENTITYSYSTEM_API void LoadAsset(Entities *entities, EntityGroup &group, const rapidxml::xml_node<lean::utf8_t> &parentNode, beCore::ParameterSet &parameters);

/// Loads entities and resources from the given xml node.
BE_ENTITYSYSTEM_API void LoadAsset(Entities *entities, EntityGroup &group, const utf8_ntri &file, beCore::ParameterSet &parameters);

} // namespace

#endif