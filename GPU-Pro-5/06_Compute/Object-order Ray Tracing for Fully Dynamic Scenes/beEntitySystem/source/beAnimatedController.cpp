/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beAnimatedController.h"
#include "beEntitySystem/beSimulation.h"

namespace beEntitySystem
{

// Constructor.
AnimatedController::AnimatedController()
{
}

// Destructor.
AnimatedController::~AnimatedController()
{
}

// Attaches this controller to the given simulation.
void AnimatedController::Attach(Simulation *simulation)
{
	simulation->AddAnimated(this);
}

// Detaches this controller from the given simulation.
void AnimatedController::Detach(Simulation *simulation)
{
	simulation->RemoveAnimated(this);
}

} // namespace
