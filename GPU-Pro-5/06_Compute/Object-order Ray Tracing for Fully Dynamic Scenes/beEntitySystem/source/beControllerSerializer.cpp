/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beControllerSerializer.h"
#include "beEntitySystem/beController.h"

#include <beCore/bePropertySerialization.h>

namespace beEntitySystem
{

// Constructor.
ControllerSerializer::ControllerSerializer(const utf8_ntri &type)
	: ComponentSerializer<Controller>(type)
{
}

// Destructor.
ControllerSerializer::~ControllerSerializer()
{
}

// Loads a controller from the given xml node.
void ControllerSerializer::Load(Controller *pController, const rapidxml::xml_node<lean::utf8_t> &node,
								beCore::ParameterSet &parameters,
								beCore::SerializationQueue<beCore::LoadJob> &queue) const
{
	ComponentSerializer<Controller>::Load(pController, node, parameters, queue);

	// Properties
	LoadProperties(*pController, node);
}

// Saves the given controller to the given XML node.
void ControllerSerializer::Save(const Controller *pController, rapidxml::xml_node<lean::utf8_t> &node,
								beCore::ParameterSet &parameters,
								beCore::SerializationQueue<beCore::SaveJob> &queue) const
{
	// TODO: Controller IDs!
//	SetID(pController->GetPersistentID(), node);
	ComponentSerializer<Controller>::Save(pController, node, parameters, queue);

	// Properties
	SaveProperties(*pController, node);
}

} // namespace