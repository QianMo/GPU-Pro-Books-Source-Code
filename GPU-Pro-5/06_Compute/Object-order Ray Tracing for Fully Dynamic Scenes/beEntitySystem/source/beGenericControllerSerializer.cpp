/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beGenericControllerSerializer.h"

namespace beEntitySystem
{

// Constructor.
AbstractGenericControllerSerializer::AbstractGenericControllerSerializer(const beCore::ComponentType *type)
	: ControllerSerializer(type->Name)
{
}

// Destructor.
AbstractGenericControllerSerializer::~AbstractGenericControllerSerializer()
{
}

// Creates a serializable object from the given parameters.
lean::scoped_ptr<Controller, lean::critical_ref> AbstractGenericControllerSerializer::Create(
	const beCore::Parameters &creationParameters, const beCore::ParameterSet &parameters) const
{
	return CreateController(parameters);
}

// Loads a mesh controller from the given xml node.
lean::scoped_ptr<Controller, lean::critical_ref> AbstractGenericControllerSerializer::Load(const rapidxml::xml_node<lean::utf8_t> &node,
	beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const
{
	lean::scoped_ptr<Controller> controller = CreateController(parameters);
	ControllerSerializer::Load(controller, node, parameters, queue);
	return controller.transfer();
}

// Saves the given mesh controller to the given XML node.
void AbstractGenericControllerSerializer::Save(const Controller *pSerializable, rapidxml::xml_node<lean::utf8_t> &node,
	beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue) const
{
	ControllerSerializer::Save(pSerializable, node, parameters, queue);
}

} // namespace
