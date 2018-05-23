/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_GENERIC_CONTROLLER_SERIALIZER
#define BE_ENTITYSYSTEM_GENERIC_CONTROLLER_SERIALIZER

#include "beEntitySystem.h"
#include "beControllerSerializer.h"

namespace beEntitySystem
{

/// Serializes mesh controllers.
class AbstractGenericControllerSerializer : public beEntitySystem::ControllerSerializer
{
protected:
	/// Creates a controller.
	virtual lean::scoped_ptr<Controller, lean::critical_ref> CreateController(const beCore::ParameterSet &parameters) const = 0;
	/// Creates a controller.
	virtual lean::scoped_ptr<Controller, lean::critical_ref> CreateController(beCore::ParameterSet &parameters) const
	{
		return CreateController( const_cast<const beCore::ParameterSet&>(parameters) );
	}

public:
	/// Constructor.
	BE_ENTITYSYSTEM_API AbstractGenericControllerSerializer(const beCore::ComponentType *type);
	/// Destructor.
	BE_ENTITYSYSTEM_API ~AbstractGenericControllerSerializer();

	/// Creates a serializable object from the given parameters.
	BE_ENTITYSYSTEM_API virtual lean::scoped_ptr<Controller, lean::critical_ref> Create(
		const beCore::Parameters &creationParameters, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE;

	/// Loads a mesh controller from the given xml node.
	BE_ENTITYSYSTEM_API virtual lean::scoped_ptr<Controller, lean::critical_ref> Load(const rapidxml::xml_node<lean::utf8_t> &node,
		beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const LEAN_OVERRIDE;
	/// Saves the given mesh controller to the given XML node.
	BE_ENTITYSYSTEM_API virtual void Save(const Controller *pSerializable, rapidxml::xml_node<lean::utf8_t> &node,
		beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue) const LEAN_OVERRIDE;
};

/// Serializes mesh controllers.
template <class Controller>
class GenericControllerSerializer : public AbstractGenericControllerSerializer
{
protected:
	/// Creates a controller.
	virtual lean::scoped_ptr<beEntitySystem::Controller, lean::critical_ref> CreateController(const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		return lean::make_scoped<Controller>();
	}

public:
	/// Constructor.
	GenericControllerSerializer()
		: AbstractGenericControllerSerializer(Controller::GetComponentType()) { }
};

} // namespace

#endif