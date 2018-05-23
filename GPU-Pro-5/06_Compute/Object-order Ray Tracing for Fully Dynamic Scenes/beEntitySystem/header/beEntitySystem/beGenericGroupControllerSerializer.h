/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_GENERIC_GROUP_CONTROLLER_SERIALIZER
#define BE_ENTITYSYSTEM_GENERIC_GROUP_CONTROLLER_SERIALIZER

#include "beEntitySystem.h"
#include "beGenericControllerSerializer.h"
#include "beSerializationParameters.h"
#include "beWorld.h"
#include "beWorldControllers.h"
#include <beCore/beParameterSet.h>

#include <lean/logging/errors.h>

namespace beEntitySystem
{

/// Serializes mesh controllers.
template <class ControllerGroup, class Derived>
class AbstractGenericGroupControllerSerializer : public AbstractGenericControllerSerializer
{
protected:
#ifdef DOXYGEN_READ_THIS
	/// Creates the controller group.
	static virtual ControllerGroup* CreateControllerGroup(World &world, beCore::ParameterSet &parameters) const = 0;
#endif
	/// Optionally gets a parameter name for quick access storage.
	LEAN_INLINE const utf8_t* ControllerGroupParameterName() const { return ControllerGroup::GetComponentType()->Name;; }
	
	/// Retrieves the controller group.
	template <class ParameterSet>
	ControllerGroup* RetrieveControllerGroup(World &world, ParameterSet &parameters) const
	{
		ControllerGroup *pGroup = world.Controllers().GetController<ControllerGroup>();
		
		if (!pGroup)
		{
			pGroup = static_cast<const Derived*>(this)->CreateControllerGroup(world, parameters);
			world.Controllers().AddControllerConsume(pGroup);
		}

		return pGroup;
	}

	/// Retrieves the controller group.
	template <class ParameterSet>
	ControllerGroup* RetrieveControllerGroup(ParameterSet &parameters) const
	{
		return RetrieveControllerGroup(
				*LEAN_THROW_NULL( beEntitySystem::GetEntitySystemParameters(parameters).World ),
				parameters
			);
	}

	struct ControllerFactory
	{
		const AbstractGenericGroupControllerSerializer *self;

		template <class ParameterSet>
		ControllerGroup* operator ()(ParameterSet &parameters, const beCore::ParameterLayout &layout) const
		{
			return self->RetrieveControllerGroup(parameters);
		}
	};

	/// Gets the controller group.
	template <class ParameterSet>
	ControllerGroup* GetControllerGroup(ParameterSet &parameters) const
	{
		if (const utf8_t* parameterName = static_cast<const Derived*>(this)->ControllerGroupParameterName())
		{
			beCore::ParameterLayout& parameterLayout = beEntitySystem::GetSerializationParameters();
			ControllerFactory factory = { this };
			return beCore::GetOrMake<ControllerGroup*, Derived>(parameters, parameterLayout, parameterName, factory);
		}
		else
			return RetrieveControllerGroup(parameters);
	}

public:
	/// Constructor.
	AbstractGenericGroupControllerSerializer(const beCore::ComponentType *type)
		: AbstractGenericControllerSerializer(type) { }
};

/// Serializes mesh controllers.
template <class Controller, class ControllerGroup, class Derived>
class GenericGroupControllerSerializer : public AbstractGenericGroupControllerSerializer<ControllerGroup, Derived>
{
protected:
	/// Creates a controller.
	virtual lean::scoped_ptr<beEntitySystem::Controller, lean::critical_ref> CreateController(beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		return lean::scoped_ptr<beEntitySystem::Controller, lean::critical_ref>(
				this->GetControllerGroup(parameters)->AddController()
			);
	}
	/// Creates a controller.
	virtual lean::scoped_ptr<beEntitySystem::Controller, lean::critical_ref> CreateController(const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		return lean::scoped_ptr<beEntitySystem::Controller, lean::critical_ref>(
				this->GetControllerGroup(parameters)->AddController()
			);
	}

public:
	/// Constructor.
	GenericGroupControllerSerializer()
		: typename GenericGroupControllerSerializer::AbstractGenericGroupControllerSerializer(Controller::GetComponentType()) { }
};

} // namespace

#endif