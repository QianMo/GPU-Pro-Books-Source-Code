/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_WORLDCONTROLLERS
#define BE_ENTITYSYSTEM_WORLDCONTROLLERS

#include "beEntitySystem.h"
#include <lean/tags/noncopyable.h>
#include <beCore/beShared.h>
#include <beCore/beComponent.h>
#include <lean/containers/simple_vector.h>
#include <beCore/beMany.h>
#include <lean/tags/move_ptr.h>

namespace beEntitySystem
{

class Entities;
class Simulation;
class WorldController;

/// Controller-driven base class.
class WorldControllers : public lean::tags::noncopyable_chain<beCore::Shared>
{
public:
	/// Controller vector type.
	typedef lean::simple_vector<WorldController*, lean::containers::vector_policies::inipod > controllers_t;

private:
	controllers_t m_controllers;

public:
	/// Controller range type.
	typedef beCore::Range<WorldController *const *> Controllers;

	/// Constructor.
	BE_ENTITYSYSTEM_API WorldControllers();
	/// Destructor.
	BE_ENTITYSYSTEM_API ~WorldControllers();

	/// Adds the given controller.
	template <class ActualType>
	LEAN_INLINE void AddController(lean::move_ptr<ActualType> controller) noexcept
	{
		AddControllerConsume(controller.peek());
		controller.transfer();
	}
	/// Adds the given controller, taking ownership.
	BE_ENTITYSYSTEM_API void AddControllerConsume(WorldController *controller) noexcept;
	/// Removes the given controller.
	BE_ENTITYSYSTEM_API void RemoveController(WorldController *controller) noexcept;
	/// Clears all controllers.
	BE_ENTITYSYSTEM_API void ClearControllers() noexcept;
	
	/// Gets a vector of all controllers.
	LEAN_INLINE Controllers GetControllers() const { return beCore::MakeRangeN(&m_controllers[0], m_controllers.size()); }
	
	/// Gets the first controller of the given type.
	BE_ENTITYSYSTEM_API WorldController* GetController(const beCore::ComponentType *type);
	/// Gets the first controller of the given type.
	template <class ControllerType>
	LEAN_INLINE ControllerType* GetController()
	{
		return dynamic_cast<ControllerType*>( GetController(ControllerType::GetComponentType()) );
	}
};

/// Attaches the given collection of simulation controllers.
BE_ENTITYSYSTEM_API void Attach(beCore::Range<WorldController *const *> controllers, Simulation *simulation);
/// Detaches the given collection of simulation controllers.
BE_ENTITYSYSTEM_API void Detach(beCore::Range<WorldController *const *> controllers, Simulation *simulation);
/// Checks for and commits all changes.
BE_ENTITYSYSTEM_API void Commit(beCore::Range<WorldController *const *> controllers);

} // namespace

#endif