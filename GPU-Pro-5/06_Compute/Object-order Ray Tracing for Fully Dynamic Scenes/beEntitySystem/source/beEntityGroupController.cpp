/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beEntityGroupController.h"
#include "beEntitySystem/beEntities.h"

#include <beCore/beReflectionProperties.h>

#include <beMath/beMatrix.h>

#include <lean/functional/algorithm.h>

#include <lean/logging/errors.h>

namespace beEntitySystem
{

BE_CORE_PUBLISH_COMPONENT(EntityGroupController)

const beCore::ReflectionProperty ControllerProperties[] =
{
	beCore::MakeReflectionProperty<bool>("move children", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&EntityGroupController::MoveChildren) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&EntityGroupController::DoesMoveChildren) )
};
BE_CORE_ASSOCIATE_PROPERTIES(EntityGroupController, ControllerProperties)

namespace
{

template <class It>
void SetOwnerWithoutNotification(It begin, It end, EntityController *owner) noexcept
{
	for (It it = begin; it < end; ++it)
		(*it)->SetOwner(owner, EntityOwnerNotification::WithoutNotification);
}

template <class It>
void UnsetOwnerWithoutNotification(It begin, It end, EntityController *owner) noexcept
{
	for (It it = begin; it < end; ++it)
		if (Entity *child = *it)
			child->UnsetOwner(owner, EntityOwnerNotification::WithoutNotification);
}

void SetOwnerWithoutNotification(EntityGroupController::entities_t &entities, EntityController *owner) noexcept
{
	SetOwnerWithoutNotification(entities.begin(), entities.end(), owner);
}

void UnsetOwnerWithoutNotification(EntityGroupController::entities_t &entities, EntityController *owner) noexcept
{
	UnsetOwnerWithoutNotification(entities.begin(), entities.end(), owner);
}

} // namespace

// Creates an empty group.
EntityGroupController::EntityGroupController()
	: m_pOwner(),
	m_bMoveChildren(true)
{
}

// Copies and entity group.
EntityGroupController::EntityGroupController(Entity *const *entities, uint4 count)
	: m_pOwner(),
	m_bMoveChildren(true)
{
	AddEntities(entities, count);
}

// Destructor.
EntityGroupController::~EntityGroupController()
{
	UnsetOwnerWithoutNotification(m_entities, this);
}

// Synchronizes this controller with the controlled entity.
void EntityGroupController::Synchronize(bees::EntityHandle entity)
{
	using bees::Entities;

	if (m_bMoveChildren)
	{
		Entities::Transformation nextTrafo = Entities::GetTransformation(entity);

		if (nextTrafo.Position != m_lastPos || nextTrafo.Orientation != m_lastOrient)
		{
			bem::fmat3 deltaOrient = mul(transpose(m_lastOrient), nextTrafo.Orientation);

			for (entities_t::iterator it = m_entities.begin(), itEnd = m_entities.end(); it < itEnd; ++it)
			{
				EntityHandle child = (*it)->Handle();
				Entities::Transformation childTrafo = Entities::GetTransformation(child);

				childTrafo.Position = mul(childTrafo.Position - m_lastPos, deltaOrient) + nextTrafo.Position;
				childTrafo.Orientation = mul(childTrafo.Orientation, deltaOrient);
			
				Entities::SetTransformation(child, childTrafo);
				Entities::NeedSync(child);
			}
		}
	}
}

// Synchronizes this controller with the controlled entity.
void EntityGroupController::Flush(const bees::EntityHandle entity)
{
	using bees::Entities;
	Entities::Transformation trafo = Entities::GetTransformation(entity);

	m_lastPos = trafo.Position;
	m_lastOrient = trafo.Orientation;
}

// Gets the centroid of the given group of entities.
bem::fvec3 GetCentroid(const Entity *const *entities, uint4 count)
{
	bem::fvec3 centroid;

	if (count)
	{
		for (const Entity *const *it = entities, *const *itEnd = entities + count; it < itEnd; ++it)
			centroid += (*it)->GetPosition();
		
		centroid /= (float) count;
	}

	return centroid;
}

// Gets the centroid of this entity group.
bem::fvec3 EntityGroupController::GetCentroid() const
{
	return bees::GetCentroid(m_entities.data(), (uint4) m_entities.size());
}

// Adds the given entities to this group.
void EntityGroupController::AddEntities(Entity *const *entities, uint4 count)
{
	LEAN_ASSERT(entities != nullptr || count == 0);

	size_t destIdx = m_entities.size();
	m_entities.resize(m_entities.size() + count);
	Entity **dest = &m_entities[destIdx];

	for (Entity *const *src = entities, *const *srcEnd = entities + count; src < srcEnd; ++src, ++dest)
		*dest = LEAN_ASSERT_NOT_NULL(*src);

	SetOwnerWithoutNotification(entities, entities + count, this);
}

// Removes the given entities from this group.
bool EntityGroupController::RemoveEntities(Entity *const *entities, uint4 count)
{
	UnsetOwnerWithoutNotification(entities, entities + count, this);

	return lean::remove_all(m_entities, entities, entities + count);
}

// Reserves space for the given number of entities.
void EntityGroupController::ReserveNextEntities(uint4 count)
{
	m_entities.reserve(m_entities.size() + count);
}

// Gets an OPTIONAL parent entity for the children of this controller.
Entity* EntityGroupController::GetParent() const
{
	return m_pOwner;
}

// Gets the rules for (child) entities owned by this controller.
uint4 EntityGroupController::GetChildFlags() const
{
	return ChildEntityFlags::OpenGroup | ChildEntityFlags::Accessible;
}

// The given child entity has been added (== owner about to be set to this).
bool EntityGroupController::ChildAdded(Entity *child)
{
	if (m_pOwner)
	{
		m_entities.push_back(LEAN_ASSERT_NOT_NULL(child));
		return true;
	}
	return false;
}

// The given child entity was removed.
bool EntityGroupController::ChildRemoved(Entity *child)
{
	if (m_pOwner)
		lean::remove(m_entities, child);
	return true;
}

// Attaches all entities to their simulations.
void EntityGroupController::Attach(Entity *entity)
{
	LEAN_ASSERT(!m_pOwner);

	bees::Attach(m_entities.data(), (uint4) m_entities.size());

	m_pOwner = entity;
	// IMPORTANT: Initialize deltas BEFORE synchronization
	EntityGroupController::Flush(entity->Handle());
}

// Detaches all entities from their simulations.
void EntityGroupController::Detach(Entity *entity)
{
	bees::Detach(m_entities.data(), (uint4) m_entities.size());

	LEAN_ASSERT(entity == m_pOwner);
	m_pOwner = nullptr;
}

// Clones this entity controller.
EntityGroupController* EntityGroupController::Clone() const
{
	size_t entityCount = m_entities.size();
	lean::scoped_ptr<lean::scoped_ptr<Entity>[] > entityClones( new lean::scoped_ptr<Entity>[entityCount] );

	// Clone entities
	for (size_t i = 0; i < entityCount; ++i)
		entityClones[i] = m_entities[i]->Clone();

	// Clone controller
	lean::scoped_ptr<EntityGroupController> clone( new EntityGroupController(&entityClones[0].get(), entityCount) );
	for (uint4 i = 0; i < entityCount; ++i) entityClones[i].detach();

	return clone.detach();
}

} // namespace
