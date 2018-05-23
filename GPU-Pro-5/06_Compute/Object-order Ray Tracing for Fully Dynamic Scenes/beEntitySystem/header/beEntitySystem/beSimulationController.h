/************************************************************/
/* breeze Engine Simulation System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_SIMULATIONCONTROLLER
#define BE_ENTITYSYSTEM_SIMULATIONCONTROLLER

#include "beEntitySystem.h"
#include <beCore/beShared.h>
#include "beController.h"

namespace beEntitySystem
{

class Simulation;

/// World controller base class.
class WorldController : public beCore::Shared, public Controller
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(WorldController)

public:
	/// Checks for and commits all changes.
	virtual void Commit() { }

	/// Attaches this controller to the given simulation.
	virtual void Attach(Simulation *simulation) { }
	/// Detaches this controller from the given simulation.
	virtual void Detach(Simulation *simulation) { }
};

/// Simulation controller base class.
class SimulationController : public WorldController
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(SimulationController)

public:
	/// Attaches this controller to the given simulation.
	virtual void Attach(Simulation *simulation) = 0;
	/// Detaches this controller from the given simulation.
	virtual void Detach(Simulation *simulation) = 0;
};

} // namespace

#endif