/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_ANIMATED
#define BE_ENTITYSYSTEM_ANIMATED

#include "beEntitySystem.h"

namespace beEntitySystem
{

/// Animated interface.
class LEAN_INTERFACE Animated
{
	LEAN_INTERFACE_BEHAVIOR(Animated)

public:
	/// Steps the animation.
	virtual void Step(float timeStep) = 0;
};

} // namespace

#endif