/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_ENTITIES
#define BE_ENTITYSYSTEM_ENTITIES

#include "beEntitySystem.h"
#include "beSynchronized.h"
#include <beCore/beShared.h>
#include <beCore/beReflectionPropertyProvider.h>

#include <lean/pimpl/static_pimpl.h>
#include <lean/smart/scoped_ptr.h>

#include <beCore/beMany.h>
#include <beMath/beVectorDef.h>
#include <beMath/beMatrixDef.h>

namespace beCore
{
	class PersistentIDs;
}

namespace beEntitySystem
{

class Entities;
class EntityController;
class Entity;

using namespace beMath::Types;

/// Handle to an entity.
struct EntityHandle : public beCore::GroupElementHandle<Entities>
{
	friend class Entities;

private:
	/// Internal constructor.
	EntityHandle(class Entities *entities, uint4 internalIdx)
		: GroupElementHandle<Entities>(entities, internalIdx) { }
};

/// Filters a collection of controllers, e.g. when entities are cloned.
class LEAN_INTERFACE EntityControllerFilter
{
	LEAN_INTERFACE_BEHAVIOR(EntityControllerFilter)

public:
	/// Return true to accept the given controller.
	virtual bool Accept(const EntityController *controller) = 0;
};

/// Ownership transferral notification flags.
struct EntityOwnerNotification
{
	enum T
	{
		WithNotification,	///< The specified owner will be called back when ownership is transferred.
		WithoutNotification	///< The specified owner will not be called back when ownership is transferred.
	};
};

/// Entity manager.
class LEAN_INTERFACE Entities : public beCore::Shared, public Synchronized,
	public beCore::UnRefCounted< beCore::RigidReflectedComponent< beCore::OptionalPropertyProvider< beCore::NoPropertyFeedbackProvider< beCore::ReflectedComponent > > > >
{
	LEAN_SHARED_SIMPL_INTERFACE_BEHAVIOR(Entities)

protected:
	// The given owned (child) entity has been removed.
	BE_ENTITYSYSTEM_API static void ChildRemoved(EntityHandle entity, Entity *child);

public:
	class M;

	/// Entity transformation.
	struct Transformation
	{
		fvec3 Position;		///< Entity position.
		fmat3 Orientation;	///< Entity orientation.
		fvec3 Scaling;		///< Entity scaling.

		/// Default constructor.
		Transformation()
			: Orientation(fmat3::identity),
			Scaling(1.0f) { }
	};

	/// Invalid persistent entity ID.
	static const uint8 InvalidPersistentID = -1;
	/// Invalid persistent entity ID.
	static const uint8 AnonymousPersistentID = InvalidPersistentID;
	/// Requests assignment of a new persistent entity ID.
	static const uint8 NewPersistentID = -2;

	/// Adds an entity.
	BE_ENTITYSYSTEM_API Entity* AddEntity(utf8_ntri name = "<unnamed>", uint8 persistentID = NewPersistentID);
	/// Clones the given entity.
	BE_ENTITYSYSTEM_API static Entity* CloneEntity(const EntityHandle entity, uint8 persistentID = NewPersistentID, EntityControllerFilter *pFilter = nullptr);
	/// Removes an entity.
	BE_ENTITYSYSTEM_API static void RemoveEntity(Entity *pEntity);
	/// Reserves space for the given number of entities.
	BE_ENTITYSYSTEM_API void Reserve(uint4 entityCount);

	/// Controller range type.
	typedef beCore::Range<Entity *const *> Range;
	/// Controller range type.
	typedef beCore::Range<const Entity *const *> ConstRange;
	/// Gets all entities.
	BE_ENTITYSYSTEM_API Range GetEntities();
	/// Gets all entities.
	BE_ENTITYSYSTEM_API ConstRange GetEntities() const;

	/// Commits changes such as addition/removal of entities and controllers.
	BE_ENTITYSYSTEM_API void Commit();
	/// Flushes entity changes.
	BE_ENTITYSYSTEM_API void Flush();
	
	/// Controller range type.
	typedef beCore::Range<EntityController *const *> Controllers;

	/// Adds the given controller.
	BE_ENTITYSYSTEM_API static void AddControllers(EntityHandle entity, EntityController *const* controllers, uint4 count, bool bConsume);
	/// Removes the given controller.
	BE_ENTITYSYSTEM_API static void RemoveControllers(EntityHandle entity, EntityController *const* controllers, uint4 count, bool bPermanently);
	/// Gets all controllers.
	BE_ENTITYSYSTEM_API static Controllers GetControllers(const EntityHandle entity);
	/// Gets the first controller of the given type.
	BE_ENTITYSYSTEM_API static EntityController* GetController(const EntityHandle entity, const beCore::ComponentType *type);
	/// Gets the first controller of the given type.
	template <class ControllerType>
	LEAN_INLINE static ControllerType* GetController(const EntityHandle entity)
	{
		return dynamic_cast<ControllerType*>( GetController(entity, ControllerType::GetComponentType()) );
	}

	/// Marks the given entity for committing.
	BE_ENTITYSYSTEM_API static void NeedCommit(EntityHandle entity);
	/// Marks the given entity for synchronization.
	BE_ENTITYSYSTEM_API static void NeedSync(EntityHandle entity);
	/// Marks the given entity for flushing.
	BE_ENTITYSYSTEM_API static void NeedFlush(EntityHandle entity);
	/// Synchronizes the given entity with its controllers.
	BE_ENTITYSYSTEM_API static void Synchronize(EntityHandle entity);

	/// Sets the position base.
	BE_ENTITYSYSTEM_API void SetPositionBase(const lvec3 &base, bool bCommit = true);
	/// Gets the position base.
	BE_ENTITYSYSTEM_API const lvec3& GetPositionBase() const;

	/// Sets the precise position.
	BE_ENTITYSYSTEM_API static void SetPrecisePosition(EntityHandle entity, const lvec3 &position);
	/// Sets the position.
	BE_ENTITYSYSTEM_API static void SetPosition(EntityHandle entity, const fvec3 &position);
	/// Sets the orientation.
	BE_ENTITYSYSTEM_API static void SetOrientation(EntityHandle entity, const fmat3 &orientation);
	/// Sets the scaling.
	BE_ENTITYSYSTEM_API static void SetScaling(EntityHandle entity, const fvec3 &scaling);
	/// Sets the transformation.
	BE_ENTITYSYSTEM_API static void SetTransformation(EntityHandle entity, const Transformation &trafo);

	/// Gets the precise position.
	BE_ENTITYSYSTEM_API static const lvec3& GetPrecisePosition(const EntityHandle entity);
	/// Gets the position.
	BE_ENTITYSYSTEM_API static const fvec3& GetPosition(const EntityHandle entity);
	/// Gets the orientation.
	BE_ENTITYSYSTEM_API static const fmat3& GetOrientation(const EntityHandle entity);
	/// Gets the scaling.
	BE_ENTITYSYSTEM_API static const fvec3& GetScaling(const EntityHandle entity);
	/// Gets the transformation.
	BE_ENTITYSYSTEM_API static const Transformation& GetTransformation(const EntityHandle entity);

	/// Sets the orientation.
	BE_ENTITYSYSTEM_API static void SetAngles(EntityHandle entity, const fvec3 &angles);
	/// Gets the orientation.
	BE_ENTITYSYSTEM_API static fvec3 GetAngles(const EntityHandle entity);

	/// Shows or hides the entity.
	BE_ENTITYSYSTEM_API static void SetVisible(EntityHandle entity, bool bVisible);
	/// Shows or hides the entity.
	BE_ENTITYSYSTEM_API static bool IsVisible(const EntityHandle entity);

	/// Sets whether the entity is serialized.
	BE_ENTITYSYSTEM_API static void SetSerialized(EntityHandle entity, bool bSerialized);
	/// Gets whether the entity is serialized.
	BE_ENTITYSYSTEM_API static bool GetSerialized(const EntityHandle entity);

	/// Sets the owner of this entity.
	BE_ENTITYSYSTEM_API static void SetOwner(EntityHandle entity, EntityController *owner, EntityOwnerNotification::T notification = EntityOwnerNotification::WithNotification);
	/// The given owner releases the entity back into the wild.
	BE_ENTITYSYSTEM_API static void UnsetOwner(EntityHandle entity, EntityController *owner, EntityOwnerNotification::T notification = EntityOwnerNotification::WithNotification);
	/// Gets the owner of this entity.
	BE_ENTITYSYSTEM_API static EntityController* GetOwner(const EntityHandle entity);

	/// Attaches the entity.
	BE_ENTITYSYSTEM_API static void Attach(EntityHandle entity);
	/// Detaches the entity.
	BE_ENTITYSYSTEM_API static void Detach(EntityHandle entity);
	/// Checks whether the given entity is attached.
	BE_ENTITYSYSTEM_API static bool IsAttached(const EntityHandle entity);

	/// Sets the name.
	BE_ENTITYSYSTEM_API static void SetName(EntityHandle entity, const utf8_ntri &name);
	/// Gets the name.
	BE_ENTITYSYSTEM_API static const utf8_string& GetName(const EntityHandle entity);

	/// Sets the persistent ID.
	BE_ENTITYSYSTEM_API static void SetPersistentID(EntityHandle entity, uint8 persistentID);
	/// Gets the persistent ID.
	BE_ENTITYSYSTEM_API static uint8 GetPersistentID(const EntityHandle entity);
	
	/// Gets the ID.
	LEAN_INLINE static uint4 GetCurrentID(const EntityHandle entity) { return entity.Index; }

	/// Sets the custom base ID.
	BE_ENTITYSYSTEM_API void SetCustomIDBase(uint4 baseID);
	/// Gets an entity from the given custom ID.
	BE_ENTITYSYSTEM_API Entity* GetEntityByCustomID(uint4 customID);
	/// Gets the custom ID.
	BE_ENTITYSYSTEM_API static uint4 GetCustomID(const EntityHandle entity);

	/// Gets the entity type.
	BE_ENTITYSYSTEM_API static const beCore::ComponentType* GetComponentType();
	/// Gets the entity type.
	BE_ENTITYSYSTEM_API const beCore::ComponentType* GetType() const;
};

/// Creates a collection of entities.
/// @relatesalso Entities
BE_ENTITYSYSTEM_API lean::scoped_ptr<Entities, lean::critical_ref> CreateEntities(beCore::PersistentIDs *persistentIDs);

/// Proxy to an entity in a group of Entities.
class Entity : public lean::noncopyable, public beCore::UnRefCounted< beCore::ReflectionPropertyProvider >
{
	friend class Entities;

private:
	EntityHandle m_handle;

	/// Internal constructor.
	Entity(EntityHandle handle)
		: m_handle(handle) { }
	~Entity() { }

public:
	/// Adds the given controller.
	template <class ActualType>
	LEAN_INLINE void AddController(lean::move_ptr<ActualType> controller) { AddControllerConsume(controller.peek()); controller.transfer(); }
	/// Adds the given controller, allowing the caller to keep the given shared reference.
	LEAN_INLINE void AddControllerKeep(EntityController *controller) { Entities::AddControllers(m_handle, &controller, 1, false); }
	/// Adds the given controller, consuming the given reference.
	LEAN_INLINE void AddControllerConsume(EntityController *controller) { Entities::AddControllers(m_handle, &controller, 1, true); }
	/// Removes the given controller.
	LEAN_INLINE void RemoveController(EntityController *controller, bool bPermanently) { Entities::RemoveControllers(m_handle, &controller, 1, bPermanently); }
	/// Adds the given controllers, allowing the caller to keep the given shared references.
	LEAN_INLINE void AddControllersKeep(EntityController *const* controllers, uint4 count) { Entities::AddControllers(m_handle, controllers, count, false); }
	/// Adds the given controllers, consuming the given references.
	LEAN_INLINE void AddControllersConsume(EntityController *const* controllers, uint4 count) { Entities::AddControllers(m_handle, controllers, count, true); }
	/// Removes the given controller.
	LEAN_INLINE void RemoveControllers(EntityController *const* controllers, uint4 count, bool bPermanently) { Entities::RemoveControllers(m_handle, controllers, count, bPermanently); }
	/// Commits changes such as addition/removal of controllers.
	LEAN_INLINE void Commit() { m_handle.Group->Commit(); }

	/// Controller range type.
	typedef Entities::Controllers Controllers;

	/// Gets all controllers.
	LEAN_INLINE Controllers GetControllers() const { return Entities::GetControllers(m_handle); }
	/// Gets the first controller of the given type.
	LEAN_INLINE EntityController* GetController(const beCore::ComponentType *type) const { return Entities::GetController(m_handle, type); }
	/// Gets the first controller of the given type.
	template <class ControllerType>
	LEAN_INLINE ControllerType* GetController() const { return Entities::GetController<ControllerType>(m_handle); }

	/// Marks this entity for committing.
	LEAN_INLINE void NeedCommit() { Entities::NeedCommit(m_handle); }
	/// Marks this entity for synchronization.
	LEAN_INLINE void NeedSync() { Entities::NeedSync(m_handle); }
	/// Marks this entity for flushing.
	LEAN_INLINE void NeedFlush() { Entities::NeedFlush(m_handle); }
	/// Synchronizes the entity with its controllers.
	LEAN_INLINE void Synchronize() { Entities::Synchronize(m_handle); }

	/// Transformation type.
	typedef Entities::Transformation Transformation;

	/// Sets the precise position.
	LEAN_INLINE void SetPrecisePosition(const lvec3 &position) { Entities::SetPrecisePosition(m_handle, position); }
	/// Sets the position.
	LEAN_INLINE void SetPosition(const fvec3 &position) { Entities::SetPosition(m_handle, position); }
	/// Sets the orientation.
	LEAN_INLINE void SetOrientation(const fmat3 &orientation) { Entities::SetOrientation(m_handle, orientation); }
	/// Sets the scaling.
	LEAN_INLINE void SetScaling(const fvec3 &scaling) { Entities::SetScaling(m_handle, scaling); }
	/// Sets the transformation.
	LEAN_INLINE void SetTransformation(const Transformation &trafo) { Entities::SetTransformation(m_handle, trafo); }

	/// Gets the precise position.
	LEAN_INLINE const lvec3& GetPrecisePosition() const { return Entities::GetPrecisePosition(m_handle); }
	/// Gets the position.
	LEAN_INLINE const fvec3& GetPosition() const { return Entities::GetPosition(m_handle); }
	/// Gets the orientation.
	LEAN_INLINE const fmat3& GetOrientation() const { return Entities::GetOrientation(m_handle); }
	/// Gets the scaling.
	LEAN_INLINE const fvec3& GetScaling() const { return Entities::GetScaling(m_handle); }
	/// Gets the transformation.
	LEAN_INLINE const Transformation& GetTransformation() const { return Entities::GetTransformation(m_handle); }

	/// Sets the orientation.
	LEAN_INLINE void SetAngles(const fvec3 &angles) { Entities::SetAngles(m_handle, angles); }
	/// Gets the orientation.
	LEAN_INLINE fvec3 GetAngles() const { return Entities::GetAngles(m_handle); }

	/// Shows or hides the entity.
	LEAN_INLINE void SetVisible(bool bVisible) { Entities::SetVisible(m_handle, bVisible); }
	/// Shows or hides the entity.
	LEAN_INLINE bool IsVisible() const { return Entities::IsVisible(m_handle); }

	/// Sets whether the entity is serialized.
	LEAN_INLINE void SetSerialized(bool bSerialized) { Entities::SetSerialized(m_handle, bSerialized); }
	/// Gets whether the entity is serialized.
	LEAN_INLINE bool IsSerialized() const { return Entities::GetSerialized(m_handle); }

	/// Sets the owner of this entity.
	LEAN_INLINE void SetOwner(EntityController *owner, EntityOwnerNotification::T notification = EntityOwnerNotification::WithNotification) { Entities::SetOwner(m_handle, owner, notification); }
	/// The given owner releases the entity back into the wild.
	LEAN_INLINE void UnsetOwner(EntityController *owner, EntityOwnerNotification::T notification = EntityOwnerNotification::WithNotification) { Entities::UnsetOwner(m_handle, owner, notification); }
	/// Gets the owner of this entity.
	LEAN_INLINE EntityController* GetOwner() const { return Entities::GetOwner(m_handle); }

	/// Attaches the entity.
	LEAN_INLINE void Attach() { Entities::Attach(m_handle); }
	/// Detaches the entity.
	LEAN_INLINE void Detach() { Entities::Detach(m_handle); }
	/// Checks whether the given entity is attached.
	LEAN_INLINE bool IsAttached() const { return Entities::IsAttached(m_handle); }

	/// Sets the name.
	LEAN_INLINE void SetName(const utf8_ntri &name) { Entities::SetName(m_handle, name); }
	/// Gets the name.
	LEAN_INLINE const utf8_string& GetName() const { return Entities::GetName(m_handle); }

	/// Sets the persistent ID.
	LEAN_INLINE void SetPersistentID(uint8 persistentID) { Entities::SetPersistentID(m_handle, persistentID); }
	/// Gets the persistent ID.
	LEAN_INLINE uint8 GetPersistentID() const { return Entities::GetPersistentID(m_handle); }

	/// Gets the ID.
	LEAN_INLINE uint4 GetCurrentID() const { return Entities::GetCurrentID(m_handle); }
	/// Gets the custom ID.
	LEAN_INLINE uint4 GetCustomID() const { return Entities::GetCustomID(m_handle); }

	/// Adds a property listener.
	BE_ENTITYSYSTEM_API void AddObserver(beCore::ComponentObserver *listener) LEAN_OVERRIDE;
	/// Removes a property listener.
	BE_ENTITYSYSTEM_API void RemoveObserver(beCore::ComponentObserver *pListener) LEAN_OVERRIDE;

	/// Hints at externally imposed changes, such as changes via an editor UI.
	BE_ENTITYSYSTEM_API void ForcedChangeHint() LEAN_OVERRIDE;

	/// Gets the reflection properties.
	BE_ENTITYSYSTEM_API static Properties GetOwnProperties();
	/// Gets the reflection properties.
	BE_ENTITYSYSTEM_API Properties GetReflectionProperties() const LEAN_OVERRIDE;

	/// Gets the entity type.
	BE_ENTITYSYSTEM_API static const beCore::ComponentType* GetComponentType();
	/// Gets the entity type.
	BE_ENTITYSYSTEM_API const beCore::ComponentType* GetType() const LEAN_OVERRIDE;

	/// Clones this entity.
	LEAN_INLINE Entity* Clone(uint8 persistentID = Entities::NewPersistentID) const { return Entities::CloneEntity(m_handle, persistentID); }
	/// Removes this entity.
	LEAN_INLINE void Abandon() const { Entities::RemoveEntity(const_cast<Entity*>(this)); }

	/// Gets the handle to the entity.
	LEAN_INLINE EntityHandle& Handle() { return m_handle; }
	/// Gets the handle to the entity.
	LEAN_INLINE const EntityHandle& Handle() const { return m_handle; }
};

/// Deletes the given entity (smart pointer compatibility).
LEAN_INLINE void release_ptr(const Entity *entity)
{
	if (entity)
		entity->Abandon();
}

/// Gets the first accessible entity (candidate or parent) for the given entity, null if none.
BE_ENTITYSYSTEM_API Entity* FirstAccessibleEntity(Entity *pCandidate);
/// Gets the next accessible parent entity for the given entity, null if none.
BE_ENTITYSYSTEM_API Entity* NextAccessibleEntity(Entity *pChild);

/// Attaches the given collection of entities.
BE_ENTITYSYSTEM_API void Attach(Entity *const *entities, uint4 count);
/// Detaches the given collection of entities.
BE_ENTITYSYSTEM_API void Detach(Entity *const *entities, uint4 count);

} // namespace

#endif