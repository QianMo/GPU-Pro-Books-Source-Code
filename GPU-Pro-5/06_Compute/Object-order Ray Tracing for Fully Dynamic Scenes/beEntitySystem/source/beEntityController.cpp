/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beEntityController.h"
#include "beEntitySystem/beEntities.h"

namespace beEntitySystem
{

// Commits large-scale changes, i.e. such affecting the structure of the scene.
void EntityController::Commit(EntityHandle entity)
{
}

// Synchronizes this controller with the given controlled entity.
void EntityController::Synchronize(EntityHandle entity)
{
}

// Synchronizes this controller with the given controlled entity.
void EntityController::Flush(const EntityHandle entity)
{
}

// Gets an OPTIONAL parent entity for the children of this controller.
Entity* EntityController::GetParent() const
{
	return nullptr;
}

// Gets the rules for (child) entities owned by this controller.
uint4 EntityController::GetChildFlags() const
{
	return ChildEntityFlags::None;
}

// The given child entity has been added.
bool EntityController::ChildAdded(Entity *entity)
{
	return false;
}

// The given child entity has been removed.
bool EntityController::ChildRemoved(Entity *entity)
{
	return false;
}

// The controller has been added to the given entity.
void EntityController::Added(Entity *entity)
{
}

// The controller has been removed from the given entity.
void EntityController::Removed(Entity *entity) noexcept
{
}

// Releases this entity controller.
void SingularEntityController::Abandon() const
{
	lean::resource_ptr<const SingularEntityController>(this, lean::bind_reference);
}

} // namespace
