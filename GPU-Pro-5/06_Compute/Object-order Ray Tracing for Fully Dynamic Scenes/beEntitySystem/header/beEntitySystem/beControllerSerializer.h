/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_CONTROLLERSERIALIZER
#define BE_ENTITYSYSTEM_CONTROLLERSERIALIZER

#include "beEntitySystem.h"
#include <beCore/beComponentSerializer.h>
#include "beController.h"

namespace beEntitySystem
{

/// Controller serializer.
class ControllerSerializer : public beCore::ComponentSerializer<Controller>
{
public:
	/// Constructor.
	BE_ENTITYSYSTEM_API ControllerSerializer(const utf8_ntri &type);
	/// Destructor.
	BE_ENTITYSYSTEM_API ~ControllerSerializer();

	// Fix overloading
	using ComponentSerializer<Controller>::Load;
	/// Loads a controller from the given xml node.
	BE_ENTITYSYSTEM_API virtual void Load(Controller *pController, const rapidxml::xml_node<lean::utf8_t> &node,
		beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const LEAN_OVERRIDE;
	/// Saves the given controller to the given XML node.
	BE_ENTITYSYSTEM_API virtual void Save(const Controller *pController, rapidxml::xml_node<lean::utf8_t> &node,
		beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue) const LEAN_OVERRIDE;
};

} // namespace

#endif