/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_ANIMATEDCONTROLLER
#define BE_ENTITYSYSTEM_ANIMATEDCONTROLLER

#include "beEntitySystem.h"
#include "beAnimatedHost.h"
#include "beSimulationController.h"

namespace beEntitySystem
{

/// Animated controller adapter.
class AnimatedController : public SimulationController, public AnimatedHost
{
public:
	/// Constructor.
	BE_ENTITYSYSTEM_API AnimatedController();
	/// Destructor.
	BE_ENTITYSYSTEM_API ~AnimatedController();

	/// Attaches this controller to the given simulation.
	BE_ENTITYSYSTEM_API void Attach(Simulation *simulation);
	/// Detaches this controller from the given simulation.
	BE_ENTITYSYSTEM_API void Detach(Simulation *simulation);
};

} // namespace

#endif