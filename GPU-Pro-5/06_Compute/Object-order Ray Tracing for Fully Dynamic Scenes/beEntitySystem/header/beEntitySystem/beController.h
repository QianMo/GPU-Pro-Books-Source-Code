/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_CONTROLLER
#define BE_ENTITYSYSTEM_CONTROLLER

#include "beEntitySystem.h"
#include <beCore/beShared.h>
#include <beCore/beExchangeContainers.h>
#include "beAttachable.h"
#include <beCore/beReflectionPropertyProvider.h>

namespace beEntitySystem
{

/// Controller interface
class LEAN_INTERFACE Controller : public beCore::UnRefCounted< beCore::NoPropertyFeedbackProvider<beCore::ReflectionPropertyProvider> >
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(Controller)

public:
	/// Gets the reflection properties.
	static Properties GetControllerProperties() { return Properties(); }
	/// Gets the reflection properties.
	Properties GetReflectionProperties() const { return Properties(); }

	/// Gets the controller type.
	virtual const beCore::ComponentType* GetType() const = 0;
};

} // namespace

#endif