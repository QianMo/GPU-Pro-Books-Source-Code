/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_ENTITY_GROUP
#define BE_ENTITYSYSTEM_ENTITY_GROUP

#include "beEntitySystem.h"
#include <lean/tags/noncopyable.h>
#include <beCore/beShared.h>
#include <lean/smart/resource_ptr.h>
#include <vector>

namespace beEntitySystem
{

// Prototypes
class Entity;

/// Entity group class.
class EntityGroup : public lean::nonassignable_chain<beCore::Shared>
{
private:
	typedef std::vector<Entity*> entity_vector;
	entity_vector m_entities;

public:
	/// Entity Range.
	typedef lean::range<Entity *const *> EntityRange;
	/// Entity Range.
	typedef lean::range<const Entity *const *> ConstEntityRange;

	/// Creates an empty group.
	BE_ENTITYSYSTEM_API EntityGroup();
	/// Copies and entity group.
	BE_ENTITYSYSTEM_API EntityGroup(const EntityGroup &right);
	/// Copies and entity group.
	BE_ENTITYSYSTEM_API EntityGroup(Entity *const *entities, uint4 count);
	/// Destructor.
	BE_ENTITYSYSTEM_API ~EntityGroup();

	/// Adds the given entity to this group.
	BE_ENTITYSYSTEM_API void AddEntity(Entity *pEntity);
	/// Removes the given entity from this group.
	BE_ENTITYSYSTEM_API bool RemoveEntity(Entity *pEntity);

	/// Gets the number of entities.
	BE_ENTITYSYSTEM_API uint4 GetEntityCount() const;
	/// Gets the n-th entity.
	LEAN_INLINE Entity* GetEntity(uint4 id)
	{
		return const_cast<Entity*>( const_cast<const EntityGroup*>(this)->GetEntity(id) );
	}
	/// Gets the n-th entity.
	BE_ENTITYSYSTEM_API const Entity* GetEntity(uint4 id) const;
	
	/// Gets a range of all entities.
	LEAN_INLINE EntityRange GetEntities()
	{
		return EntityRange( &m_entities[0], &m_entities[0] + m_entities.size() );
	}
	/// Gets a range of all entities.
	LEAN_INLINE ConstEntityRange GetEntities() const
	{
		return ConstEntityRange( &m_entities[0], &m_entities[0] + m_entities.size() );
	}

	/// Attaches all entities to their simulations.
	BE_ENTITYSYSTEM_API void Attach() const;
	/// Detaches all entities from their simulations.
	BE_ENTITYSYSTEM_API void Detach() const;
};

} // namespace

#endif