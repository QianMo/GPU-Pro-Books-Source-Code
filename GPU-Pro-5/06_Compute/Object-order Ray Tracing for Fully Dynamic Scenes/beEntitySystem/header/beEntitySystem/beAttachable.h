/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_ATTACHABLE
#define BE_ENTITYSYSTEM_ATTACHABLE

#include "beEntitySystem.h"

namespace beEntitySystem
{

/// Attachable interface.
class LEAN_INTERFACE Attachable
{
public:
	/// Attaches this attachable object.
	virtual void Attach() = 0;
	/// Detaches this attachable object.
	virtual void Detach() = 0;
};

/// Attachable interface.
template <class ControllerDriven>
class LEAN_INTERFACE AttachableController
{
public:
	/// Attaches this attachable object.
	virtual void Attach(ControllerDriven *pControllerDriven) = 0;
	/// Detaches this attachable object.
	virtual void Detach(ControllerDriven *pControllerDriven) = 0;
};

} // namespace

#endif