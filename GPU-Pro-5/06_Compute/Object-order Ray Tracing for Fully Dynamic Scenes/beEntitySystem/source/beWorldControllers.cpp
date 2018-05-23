/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beWorldControllers.h"
#include "beEntitySystem/beSimulationController.h"
#include <lean/smart/scoped_ptr.h>
#include <lean/functional/algorithm.h>
#include <lean/logging/errors.h>

namespace beEntitySystem
{

// Constructor.
WorldControllers::WorldControllers()
{
}

// Destructor.
WorldControllers::~WorldControllers()
{
	ClearControllers();
}

// Adds the given controller.
void WorldControllers::AddControllerConsume(WorldController *controller) noexcept
{
	if (!controller)
	{
		LEAN_LOG_ERROR_MSG("controller may not be nullptr");
		return;
	}

	lean::push_unique(m_controllers, controller);
}

// Removes the given controller.
void WorldControllers::RemoveController(WorldController *controller) noexcept
{
	lean::remove(m_controllers, controller);
}

// Gets the first controller of the given type.
WorldController* WorldControllers::GetController(const beCore::ComponentType *type)
{
	for (controllers_t::const_iterator it = m_controllers.begin(); it != m_controllers.end(); ++it)
		if ((*it)->GetType() == type)
			return *it;

	return nullptr;
}

// Clears all controllers.
void WorldControllers::ClearControllers() noexcept
{
	// NOTE: Controllers are ONLY deleted when they are still registered here
	for (controllers_t::const_iterator it = m_controllers.end(); it-- > m_controllers.begin(); )
		delete *it;

	m_controllers.clear();
}

// Attaches the given collection of simulation controllers.
void Attach(beCore::Range<WorldController *const *> controllers, Simulation *simulation)
{
	beCore::Range<WorldController *const *> attached = beCore::MakeRangeN(controllers.Begin, 0);

	try
	{
		for (; attached.End < controllers.End; ++attached.End)
			(*attached.End)->Attach(simulation);
	}
	catch (...)
	{
		Detach(attached, simulation);
		throw;
	}
}

// Detaches the given collection of simulation controllers.
void Detach(beCore::Range<WorldController *const *> controllers, Simulation *simulation)
{
	for (; controllers.End-- > controllers.Begin; )
		(*controllers.End)->Detach(simulation);
}

/// Checks for and commits all changes.
void Commit(beCore::Range<WorldController *const *> controllers)
{
	for (; controllers.Begin < controllers.End; ++controllers.Begin)
			(*controllers.Begin)->Commit();
}

} // namespace
