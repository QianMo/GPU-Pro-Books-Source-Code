/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beEntities.h"
#include "beEntitySystem/beEntityController.h"

#include <lean/containers/simple_vector.h>
#include <lean/containers/multi_vector.h>
#include <lean/memory/chunk_pool.h>

#include <beCore/beReflectionProperties.h>
#include <beCore/beSpecialReflectionProperties.h>
#include <beCore/bePersistentIDs.h>

#include <beMath/beMatrix.h>

#include <lean/logging/errors.h>
#include <lean/functional/algorithm.h>

#define TO_FLOAT_POSITION 1.0e-3f
#define FROM_FLOAT_POSITION 1.0e3f

namespace beEntitySystem
{

BE_CORE_PUBLISH_COMPONENT(Entity)
BE_CORE_PUBLISH_COMPONENT(Entities)

const bec::ReflectionProperty EntityProperties[] =
{
	bec::MakeReflectionProperty<int8[3]>("precise pos", bec::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&Entity::SetPrecisePosition, long long) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&Entity::GetPrecisePosition, long long) ),
	bec::MakeReflectionProperty<float[3]>("position", bec::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&Entity::SetPosition, float) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&Entity::GetPosition, float) ),
	bec::MakeReflectionProperty<float[9]>("orientation", bec::Widget::None, bec::PropertyPersistence::None)
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&Entity::SetOrientation, float) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&Entity::GetOrientation, float) ),
	bec::MakeReflectionProperty<float[3]>("angles (zxy)", bec::Widget::Angle)
		.set_setter( bec::MakeEulerZXYMatrixSetter( BE_CORE_PROPERTY_SETTER(&Entity::SetOrientation) ) )
		.set_getter( bec::MakeEulerZXYMatrixGetter( BE_CORE_PROPERTY_GETTER(&Entity::GetOrientation) ) ),
	// Legacy euler angles w/ 'wrong' order of application!
	bec::MakeReflectionProperty<float[3]>("angles", bec::Widget::Angle, bec::PropertyPersistence::Read)
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&Entity::SetAngles, float) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&Entity::GetAngles, float) ),
	bec::MakeReflectionProperty<float[3]>("scaling", bec::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&Entity::SetScaling, float) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&Entity::GetScaling, float) ),
	bec::MakeReflectionProperty<bool>("visible", bec::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&Entity::SetVisible) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&Entity::IsVisible) ),
	bec::MakeReflectionProperty<uint8>("id", bec::Widget::Raw, bec::PropertyPersistence::None)
		.set_getter( BE_CORE_PROPERTY_GETTER(&Entity::GetPersistentID) )
};
BE_CORE_ASSOCIATE_PROPERTIES(Entity, EntityProperties)

class Entities::M : public Entities
{
public:
	beCore::PersistentIDs *persistentIDs;

	struct Registry
	{
		utf8_string Name;
		uint8 PersistentID;
		EntityController *pOwner;

		Registry(utf8_ntri name, uint8 persistentID)
			: Name(name.to<utf8_string>()),
			PersistentID(persistentID),
			pOwner(nullptr) { }
	};

	typedef beCore::Range<uint4> EntityControllers;

	struct PreciseTransformation
	{
		lvec3 PrecisePos;
	};

	struct State
	{
		bool Attached : 1;
		bool Visible : 1;
		bool Serialized : 1;

		State()
			: Attached(true),
			Visible(true),
			Serialized(true) { }
	};

	struct ChangedFlags
	{
		enum T
		{
			None = 0x0,
			Unchanged = 0x0,
			NeedsFlush = 0x1,
			NeedsSync = 0x2
		};
	};

	enum registry_tag { registry };
	enum reflected_tag { reflected };
	enum observers_tag { observers };
	enum controllers_tag { controllers };
	enum preciseTransformation_tag { preciseTransformation };
	enum transformation_tag { transformation };
	enum state_tag { state };
	enum changedFlags_tag { changedFlags };

	typedef lean::chunk_pool<Entity, 128> handle_pool;
	handle_pool handles;

	typedef std::vector<EntityController*> controller_vector;
	controller_vector controllerPool;

	typedef lean::multi_vector_t< lean::simple_vector_binder<lean::vector_policies::semipod> >::make<
			Registry, registry_tag,
			Entity*, reflected_tag,
			EntityControllers, controllers_tag,
			PreciseTransformation, preciseTransformation_tag,
			Transformation, transformation_tag,
			State, state_tag,
			uint1, changedFlags_tag,
			bec::ComponentObserverCollection, observers_tag
		>::type entities_t;
	entities_t entities;

	uint4 customBaseID;
	
	lvec3 positionBase;

	typedef std::vector<uint4> id_vector; 
	typedef std::vector<Entity*> entity_vector; 

	template <class Element>
	struct ChangeList
	{
		typedef std::vector<Element> vector; 
		vector collect, process;
		bool all;

		ChangeList() : all() { }
		
		void InvalidateIDs()
		{
			all = !collect.empty();
			collect.clear();
		}
		bool NextBatch()
		{
			if (process.empty())
			{
				swap(collect, process);
				return !process.empty();
			}
			else
				return false;
		}
		void DiscardBatch()
		{
			process.clear();
		}
	};
	ChangeList<Entity*> commitList;
	ChangeList<uint4> syncList;
	ChangeList<uint4> flushList;

	lvec3 nextPositionBase;
	bool positionBaseChanged;

	bool inDestruction;

	M(beCore::PersistentIDs *persistentIDs)
		: persistentIDs( LEAN_ASSERT_NOT_NULL(persistentIDs) ),
		customBaseID(0),
		positionBase(0),
		positionBaseChanged(false),
		inDestruction(false) { }
	~M()
	{
		inDestruction = true;

		// ASSERT: Entity & controller management mutation disabled apart from controller ranges

		for (uint4 entityIdx = 0, entityCount = (uint4) entities.size(); entityIdx < entityCount; ++entityIdx)
			// NOTE: Anything may happen here, refetch EVERYTHING
			for (EntityControllers &controllers = entities(M::controllers)[entityIdx]; controllers; )
				controllerPool[--controllers.End]->Release();
	}

	/// Gets the number of child components.
	uint4 GetComponentCount() const
	{
		return static_cast<uint4>(entities.size());
	}
	/// Gets the name of the n-th child component.
	beCore::Exchange::utf8_string GetComponentName(uint4 idx) const
	{
		beCore::Exchange::utf8_string result;

		if (idx < entities.size())
			result.assign(entities[idx].Name.begin(), entities[idx].Name.end());

		return result;
	}
	/// Gets the n-th reflected child component, nullptr if not reflected.
	lean::com_ptr<const ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx) const LEAN_OVERRIDE
	{
		return (idx < entities.size()) ? entities(reflected)[idx] : nullptr;
	}

	/// Makes an entity handle.
	static LEAN_INLINE EntityHandle MakeHandle(Entities *entities, uint4 internalIdx) { return EntityHandle(entities, internalIdx); }
	/// Verifies the given handle.
	friend LEAN_INLINE bool VerifyHandle(const M &m, const EntityHandle handle) { return handle.Index < m.entities.size(); }
};

namespace
{

void ScheduleFlush(Entities::M &m, uint4 internalIdx)
{
	LEAN_FREE_PIMPL(Entities);

	uint1 &changed = m.entities(M::changedFlags)[internalIdx];
	if (!changed)
	{
		m.flushList.collect.push_back(internalIdx);
		changed = M::ChangedFlags::NeedsFlush;
	}
	else
		// ASSERT: Any change implies flush
		LEAN_ASSERT(changed & M::ChangedFlags::NeedsFlush);
}

void ScheduleSync(Entities::M &m, uint4 internalIdx)
{
	LEAN_FREE_PIMPL(Entities);

	uint1 &changed = m.entities(M::changedFlags)[internalIdx];
	if (!(changed & M::ChangedFlags::NeedsSync))
	{
		// NOTE: Synchronization implies flush
		ScheduleFlush(m, internalIdx);
		m.syncList.collect.push_back(internalIdx);
		changed |= M::ChangedFlags::NeedsSync;
	}
}

} // namespace

// Creates a collection of entities.
lean::scoped_ptr<Entities, lean::critical_ref> CreateEntities(beCore::PersistentIDs *persistentIDs)
{
	return new_scoped Entities::M(persistentIDs);
}

// Adds an entity
Entity* Entities::AddEntity(utf8_ntri name, uint8 persistentID)
{
	LEAN_STATIC_PIMPL();
	LEAN_ASSERT(!m.inDestruction);

	if (persistentID == NewPersistentID)
		persistentID = m.persistentIDs->ReserveID();

	// Create tracking handle
	uint4 internalIdx = static_cast<uint4>(m.entities.size());
	Entity *handle = new(m.handles.allocate()) Entity( EntityHandle(&m, internalIdx) );

	try
	{
		// Register persistent entity
		if (persistentID != AnonymousPersistentID && !m.persistentIDs->SetReference(persistentID, handle, true))
			LEAN_THROW_ERROR_CTX("Persistent entity ID collision", name.c_str());

		try
		{
			// Insert entity data
			uint4 endOfControllersIdx = static_cast<uint4>(m.controllerPool.size());
			m.entities.push_back(
					M::Registry(name, persistentID),
					handle,
					M::EntityControllers(endOfControllersIdx, endOfControllersIdx)
				);
		}
		catch (...)
		{
			m.persistentIDs->UnsetReference(persistentID, handle);
			throw;
		}
	}
	catch (...)
	{
		m.handles.free(handle);
		throw;
	}

	// Persistent entities serialized by default
	m.entities(M::state)[internalIdx].Serialized = (persistentID != AnonymousPersistentID);

	return handle;
}

// Clones the given entity.
Entity* Entities::CloneEntity(const EntityHandle entity, uint8 persistentID, EntityControllerFilter *pFilter)
{
	BE_STATIC_PIMPL_HANDLE(const_cast<EntityHandle&>(entity));
	LEAN_ASSERT(!m.inDestruction);

	const M::EntityControllers entityControllers = m.entities(M::controllers)[entity.Index];
	uint4 controllerCount = Size4(entityControllers);
	lean::scoped_ptr<lean::scoped_ptr<EntityController>[] > controllerClones( new lean::scoped_ptr<EntityController>[controllerCount] );

	// Clone controllers
	controllerCount = 0;
	for (uint4 controllerIdx = entityControllers.Begin; controllerIdx < entityControllers.End; ++controllerIdx)
	{
		const EntityController *sourceController = m.controllerPool[controllerIdx];
		
		if (!pFilter || pFilter->Accept(sourceController))
		{
			controllerClones[controllerCount] = sourceController->Clone();
			++controllerCount;
		}
	}

	// Clone entity
	lean::scoped_ptr<Entity> clone( m.AddEntity(m.entities[entity.Index].Name, persistentID) );
	m.entities(M::preciseTransformation)[clone->Handle().Index] = m.entities(M::preciseTransformation)[entity.Index];
	m.entities(M::transformation)[clone->Handle().Index] = m.entities(M::transformation)[entity.Index];

	// Add cloned controllers
	AddControllers(clone->Handle(), &controllerClones[0].get(), controllerCount, true);
	for (uint4 i = 0; i < controllerCount; ++i) controllerClones[i].detach();

	return clone.detach();
}

// Removes an entity.
void Entities::RemoveEntity(Entity *pEntity)
{
	if (!pEntity || !pEntity->Handle().Group)
		return;

	EntityHandle entity = pEntity->Handle();
	BE_STATIC_PIMPL_HANDLE(entity);

	{
		const M::Registry &reg = m.entities[entity.Index];

		if (reg.pOwner)
			SetOwner(entity, nullptr);

		// Remove persistent entity
		m.persistentIDs->UnsetReference(reg.PersistentID, pEntity);
	}

	// Detach & remove controllers first
	RemoveControllers(entity, nullptr, 0, true);
	LEAN_ASSERT(Size(m.entities(M::controllers)[entity.Index]) == 0);

	// Keep destruction simple & linear
	if (!m.inDestruction)
	{
		m.entities.erase(entity.Index);
		// NOTE: Does not throw
		m.handles.free(pEntity);

		// Fix subsequent handles
		for (uint4 internalIdx = entity.Index, entityCount = (uint4) m.entities.size(); internalIdx < entityCount; ++internalIdx)
			m.entities(M::reflected)[internalIdx]->Handle().SetIndex(internalIdx);

		// IDs no longer match entities
		m.syncList.InvalidateIDs();
		m.flushList.InvalidateIDs();
		lean::remove(m.commitList.process, pEntity);
	}
}

// Reserves space for the given number of entities.
void Entities::Reserve(uint4 entityCount)
{
	LEAN_STATIC_PIMPL();
	m.handles.reserve(entityCount);
	m.entities.reserve(entityCount);
	m.controllerPool.reserve(entityCount + entityCount / 2);
}

// Gets all entities.
Entities::Range Entities::GetEntities()
{
	LEAN_STATIC_PIMPL();
	return beCore::MakeRangeN<Range::index_type>(m.entities(M::reflected).data(), m.entities.size());
}

// Gets all entities.
Entities::ConstRange Entities::GetEntities() const
{
	LEAN_STATIC_PIMPL_CONST();
	return beCore::MakeRangeN<ConstRange::index_type>(m.entities(M::reflected).data(), m.entities.size());
}

namespace
{

void RevertControllers(Entities::M &m, uint4 entityIdx, uint4 revertCount) noexcept
{
	LEAN_FREE_PIMPL(Entities);

	// Remove given number of controllers
	M::EntityControllers &entityControllers = m.entities(M::controllers)[entityIdx];
	m.controllerPool.erase(m.controllerPool.begin() + entityControllers.End - revertCount, m.controllerPool.begin() + entityControllers.End);
	entityControllers.End -= revertCount;

	// Fix subsequent controller ranges
	for (uint4 internalIdx = entityIdx + 1, entityCount = (uint4) m.entities.size(); internalIdx < entityCount; ++internalIdx)
	{
		m.entities(M::controllers)[internalIdx].Begin -= revertCount;
		m.entities(M::controllers)[internalIdx].End -= revertCount;
	}
}

} // namespace

// Adds the given controller.
void Entities::AddControllers(EntityHandle entity, EntityController *const* controllers, uint4 count, bool bConsume)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	LEAN_ASSERT(!m.inDestruction);

	// Add controllers to entity controller range
	M::EntityControllers &entityControllers = m.entities(M::controllers)[entity.Index];
	m.controllerPool.insert(m.controllerPool.begin() + entityControllers.End, controllers, controllers + count);
	entityControllers.End += count;

	// Fix subsequent controller ranges
	for (uint4 internalIdx = entity.Index + 1, entityCount = (uint4) m.entities.size(); internalIdx < entityCount; ++internalIdx)
	{
		m.entities(M::controllers)[internalIdx].Begin += count;
		m.entities(M::controllers)[internalIdx].End += count;
	}
	
	// Addition logic
	{
		Entity *handle = m.entities(M::reflected)[entity.Index];

		// Logically add controllers
		{
			uint4 addedCount = 0;

			try
			{
				for (; addedCount < count; ++addedCount)
					controllers[addedCount]->Added(handle);
			}
			catch (...)
			{
				for (uint4 i = addedCount; i-- > 0; )
					controllers[i]->Removed(handle);
				RevertControllers(m, entity.Index, count);
				throw;
			}
		}

		// Logically attach controllers
		if (m.entities(M::state)[entity.Index].Attached)
		{
			uint4 addedCount = 0;

			try
			{
				for (; addedCount < count; ++addedCount)
					controllers[addedCount]->Attach(handle);
			}
			catch (...)
			{
				for (uint4 i = addedCount; i-- > 0; )
					controllers[i]->Detach(handle);
				RevertControllers(m, entity.Index, count);
				throw;
			}
		}
	}

	// Final treatment; Controllers logically added, MAY NOT THROW
	try
	{
		if (!bConsume)
		{
			for (uint4 addedCount = 0; addedCount < count; ++addedCount)
				controllers[addedCount]->AddRef();
		}

		ScheduleSync(m, entity.Index);
	}
	LEAN_ASSERT_NOEXCEPT
}

// Removes the given controller.
void Entities::RemoveControllers(EntityHandle entity, EntityController *const* removeControllers, uint4 removeControllerCount, bool bPermanently)
{
	BE_STATIC_PIMPL_HANDLE(entity);

	Entity *handle = m.entities(M::reflected)[entity.Index];
	M::EntityControllers &entityControllers = m.entities(M::controllers)[entity.Index];
	bool bWasAttached = m.entities(M::state)[entity.Index].Attached;

	uint4 removedCount = 0;

	try
	{
		// Remove all controllers
		if (!removeControllers)
		{
			uint4 controllerRangeEnd = entityControllers.End;

			// Logically detach and remove in reverse order
			for (uint4 controllerIdx = entityControllers.End; controllerIdx-- > entityControllers.Begin; )
			{
				if (bWasAttached)
					m.controllerPool[controllerIdx]->Detach(handle);

				m.controllerPool[controllerIdx]->Removed(handle);
				--entityControllers.End;

				if (bPermanently)
					m.controllerPool[controllerIdx]->Abandon();
			}
			// NOTE: Controller range kept up-to-date throughout entire remove operation
			LEAN_ASSERT(entityControllers.Begin == entityControllers.End);

			// Keep destruction simple & linear
			if (!m.inDestruction)
			{
				// Actually erase controller range
				m.controllerPool.erase(m.controllerPool.begin() + entityControllers.Begin, m.controllerPool.begin() + controllerRangeEnd);
				removedCount = controllerRangeEnd - entityControllers.Begin;
			}
		}
		// Remove the given controllers
		else
		{
			// Remove in reverse order
			for (uint4 controllerIdx = entityControllers.End; controllerIdx-- > entityControllers.Begin; )
				// Compare to all given controllers
				for (uint4 removeIdx = 0; removeIdx < removeControllerCount; ++removeIdx)
					if (m.controllerPool[controllerIdx] == removeControllers[removeIdx])
					{
						// Logically detach & remove controller
						if (bWasAttached)
							removeControllers[removeIdx]->Detach(handle);

						removeControllers[removeIdx]->Removed(handle);
						--entityControllers.End;

						if (bPermanently)
							removeControllers[removeIdx]->Abandon();

						// Keep destruction simple & linear
						if (!m.inDestruction)
						{
							// Actually erase controller & keep controller range up-to-date
							m.controllerPool.erase(m.controllerPool.begin() + controllerIdx);
							++removedCount;
						}

						break;
					}
		}
	}
	LEAN_ASSERT_NOEXCEPT;

	// Keep destruction simple & linear
	if (!m.inDestruction)
		// Fix subsequent controller ranges
		for (uint4 internalIdx = entity.Index + 1, entityCount = (uint4) m.entities.size(); internalIdx < entityCount; ++internalIdx)
		{
			m.entities(M::controllers)[internalIdx].Begin -= removedCount;
			m.entities(M::controllers)[internalIdx].End -= removedCount;
		}
}

// Gets all controllers.
Entities::Controllers Entities::GetControllers(const EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE_CONST(entity);

	const M::EntityControllers &controllers = m.entities(M::controllers)[entity.Index];
	return Entities::Controllers(&m.controllerPool[controllers.Begin], &m.controllerPool[controllers.End]);
}

// Gets the first controller of the given type.
EntityController* Entities::GetController(const EntityHandle entity, const beCore::ComponentType *type)
{
	BE_STATIC_PIMPL_HANDLE_CONST(entity);

	const M::EntityControllers &controllers = m.entities(M::controllers)[entity.Index];

	for (uint4 controllerIdx = controllers.Begin; controllerIdx < controllers.End; ++controllerIdx)
		if(m.controllerPool[controllerIdx]->GetType() == type)
			return m.controllerPool[controllerIdx];

	return nullptr;
}

namespace
{

void PropertyChanged(const Entities::M &m, uint4 internalIdx)
{
	LEAN_FREE_PIMPL(Entities);
	const bec::ComponentObserverCollection &observers = m.entities(M::observers)[internalIdx];

	if (observers.HasObservers())
		observers.EmitPropertyChanged(*m.entities(M::reflected)[internalIdx]);
}

LEAN_INLINE fvec3 FromPrecisePosition(const lvec3 &precise, const lvec3 &base)
{
	return fvec3(precise - base) * TO_FLOAT_POSITION;
}

LEAN_INLINE lvec3 ToPrecisePosition(const fvec3 &floating, const lvec3 &base)
{
	return lvec3(floating * FROM_FLOAT_POSITION) + base;
}

/// Applies the new base position to all entities.
void ApplyBasePosition(Entities::M &m)
{
	LEAN_FREE_PIMPL(Entities);

	if (m.positionBaseChanged)
	{
		// TODO: Threading -> double buffer

		const size_t entityCount = m.entities.size();

		for (size_t i = 0; i < entityCount; ++i)
			// Update floating-point position
			m.entities(M::transformation)[i].Position = FromPrecisePosition(m.entities(M::preciseTransformation)[i].PrecisePos, m.nextPositionBase);

		m.positionBase = m.nextPositionBase;
		m.positionBaseChanged = false;

		m.syncList.all = true;
		m.flushList.all = true;
	}
}

} // namespace

// Sets the position base.
void Entities::SetPositionBase(const lvec3 &base, bool bCommit)
{
	LEAN_STATIC_PIMPL();
	
	m.nextPositionBase = base;
	m.positionBaseChanged = true;

	if (bCommit)
		ApplyBasePosition(m);
}

// Gets the position base.
const lvec3& Entities::GetPositionBase() const
{
	LEAN_STATIC_PIMPL_CONST();
	return m.positionBase;
}

// Commits changes such as addition/removal of entities and controllers.
void Entities::Commit()
{
	LEAN_STATIC_PIMPL();

	m.commitList.NextBatch();

	// Process current batch of changed entities
	for (M::entity_vector::iterator itChanged = m.commitList.process.begin(), itChangedEnd = m.commitList.process.end();
		itChanged != itChangedEnd; ++itChanged)
	{
		Entity *entity = *itChanged;

		EntityHandle handle = entity->Handle();
		M::EntityControllers controllers = m.entities(M::controllers)[handle.Index];

		for (uint4 controllerIdx = 0; controllerIdx < Size4(controllers);
			handle = entity->Handle(), controllers = m.entities(M::controllers)[handle.Index], ++controllerIdx)
			m.controllerPool[controllers.Begin + controllerIdx]->Commit(handle);
	}

	m.commitList.DiscardBatch();
}

namespace
{

template <void (EntityController::*ControllerCall)(EntityHandle)>
LEAN_INLINE void ProcessChanges(Entities *entities, Entities::M::ChangeList<uint4> &changeList)
{
	LEAN_FREE_STATIC_PIMPL_AT(Entities, *entities);

	// IMPORTANT: No structural changes allowed during flush!
	// TODO: Enforce in debug?
	const uint4 entityCount = (uint4) m.entities.size();
	const M::EntityControllers *controllersBegin = m.entities(M::controllers).data();
	EntityController *const *controllerPoolBegin = m.controllerPool.data();

	if (changeList.all)
	{
		// ORDER: Reset straight away to not miss subsequent update requests
		changeList.all = false;

		// Update all entities
		for (uint4 internalIdx = 0; internalIdx < entityCount; ++internalIdx)
		{
			const M::EntityControllers &controllers = controllersBegin[internalIdx];

			for (uint4 controllerIdx = controllers.Begin; controllerIdx < controllers.End; ++controllerIdx)
				(controllerPoolBegin[controllerIdx]->*ControllerCall)( M::MakeHandle(entities, internalIdx) );
		}
	}
	else
	{
		// Process current batch of changed entities
		for (M::id_vector::iterator itChanged = changeList.process.begin(), itChangedEnd = changeList.process.end();
			itChanged != itChangedEnd; ++itChanged)
		{
			uint4 internalIdx = *itChanged;
			const M::EntityControllers &controllers = controllersBegin[internalIdx];

			for (uint4 controllerIdx = controllers.Begin; controllerIdx < controllers.End; ++controllerIdx)
				(controllerPoolBegin[controllerIdx]->*ControllerCall)( M::MakeHandle(entities, internalIdx) );
		}
	}
}

} // namespace

// Flushes entity changes.
void Entities::Flush()
{
	LEAN_STATIC_PIMPL();

	ApplyBasePosition(m);

	// Next batch of changes
	bool bNewSyncBatch = m.syncList.NextBatch();
	bool bNewFlushBatch = m.flushList.NextBatch();
	if (bNewSyncBatch || bNewFlushBatch)
		memset( m.entities(M::changedFlags).data(), 0, sizeof(m.entities(M::changedFlags)[0]) * m.entities(M::changedFlags).size() );

	ProcessChanges<&EntityController::Synchronize>(this, m.syncList);
	m.syncList.DiscardBatch();

	ProcessChanges<&EntityController::Flush>(this, m.flushList);
	m.flushList.DiscardBatch();
}

// Marks the given entity for synchronization.
void Entities::NeedSync(EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	ScheduleSync(m, entity.Index);
}

// Marks the given entity for flushing.
void Entities::NeedFlush(EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	ScheduleFlush(m, entity.Index);
}

// Marks the given entity for committing.
void Entities::NeedCommit(EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	lean::push_unique(m.commitList.collect, m.entities(M::reflected)[entity.Index]);
}

// Synchronizes the given entity with its controllers.
void Entities::Synchronize(EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	const M::EntityControllers &controllers = m.entities(M::controllers)[entity.Index];

	// NOTE: Synchronization always implies flush!
	for (uint4 controllerIdx = controllers.Begin; controllerIdx < controllers.End; ++controllerIdx)
		m.controllerPool[controllerIdx]->Synchronize(entity);
	for (uint4 controllerIdx = controllers.Begin; controllerIdx < controllers.End; ++controllerIdx)
		m.controllerPool[controllerIdx]->Flush(entity);
}

// Sets the (cell-relative) position.
void Entities::SetPrecisePosition(EntityHandle entity, const lvec3 &position)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	m.entities(M::preciseTransformation)[entity.Index].PrecisePos = position;
	
	// Update floating-point position
	m.entities(M::transformation)[entity.Index].Position = FromPrecisePosition(position, m.positionBase);

	ScheduleFlush(m, entity.Index);
	PropertyChanged(m, entity.Index);
}
// Sets the (cell-relative) position.
void Entities::SetPosition(EntityHandle entity, const fvec3 &position)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	m.entities(M::transformation)[entity.Index].Position = position;

	// Update precise position
	m.entities(M::preciseTransformation)[entity.Index].PrecisePos = ToPrecisePosition(position, m.positionBase);

	ScheduleFlush(m, entity.Index);
	PropertyChanged(m, entity.Index);
}
// Sets the orientation.
void Entities::SetOrientation(EntityHandle entity, const fmat3 &orientation)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	m.entities(M::transformation)[entity.Index].Orientation = orientation;

	ScheduleFlush(m, entity.Index);
	PropertyChanged(m, entity.Index);
}
// Sets the scaling.
void Entities::SetScaling(EntityHandle entity, const fvec3 &scaling)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	m.entities(M::transformation)[entity.Index].Scaling = scaling;

	ScheduleFlush(m, entity.Index);
	PropertyChanged(m, entity.Index);
}
// Sets the transformation.
void Entities::SetTransformation(EntityHandle entity, const Transformation &trafo)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	m.entities(M::transformation)[entity.Index] = trafo;

	// Update precise position
	m.entities(M::preciseTransformation)[entity.Index].PrecisePos = ToPrecisePosition(trafo.Position, m.positionBase);

	ScheduleFlush(m, entity.Index);
	PropertyChanged(m, entity.Index);
}

// Gets the cell.
const lvec3& Entities::GetPrecisePosition(const EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE_CONST(entity);
	return m.entities(M::preciseTransformation)[entity.Index].PrecisePos;
}
// Gets the transformation.
const Entities::Transformation& Entities::GetTransformation(const EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE_CONST(entity);
	return m.entities(M::transformation)[entity.Index];
}
// Gets the (cell-relative) position.
const fvec3& Entities::GetPosition(const EntityHandle entity)
{
	return GetTransformation(entity).Position;
}
// Gets the orientation.
const fmat3& Entities::GetOrientation(const EntityHandle entity)
{
	return GetTransformation(entity).Orientation;
}
// Gets the scaling.
const fvec3& Entities::GetScaling(const EntityHandle entity)
{
	return GetTransformation(entity).Scaling;
}

// Sets the orientation.
void Entities::SetAngles(EntityHandle entity, const fvec3 &angles)
{
	SetOrientation(
			entity,
			beMath::mat_rot_yxz<3>(
				angles[0] * beMath::Constants::degrees<float>::deg2rad,
				angles[1] * beMath::Constants::degrees<float>::deg2rad,
				angles[2] * beMath::Constants::degrees<float>::deg2rad
			)
		);
}
// Gets the orientation.
fvec3 Entities::GetAngles(const EntityHandle entity)
{
	return angles_rot_yxz(GetOrientation(entity)) * beMath::Constants::degrees<float>::rad2deg;
}

/// Shows or hides the entity.
void Entities::SetVisible(EntityHandle entity, bool bVisible)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	M::State &state = m.entities(M::state)[entity.Index];

	if (state.Visible != bVisible)
	{
		state.Visible = bVisible;

		ScheduleFlush(m, entity.Index);
		PropertyChanged(m, entity.Index);
	}
}
/// Shows or hides the entity.
bool Entities::IsVisible(const EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE_CONST(entity);
	return m.entities(M::state)[entity.Index].Visible;
}

// Sets whether the entity is serialized.
void Entities::SetSerialized(EntityHandle entity, bool bSerialized)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	m.entities(M::state)[entity.Index].Serialized = bSerialized;
}

// Gets whether the entity is serialized.
bool Entities::GetSerialized(const EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE_CONST(entity);
	return m.entities(M::state)[entity.Index].Serialized;
}

// The given owned (child) entity has been removed.
void Entities::ChildRemoved(EntityHandle entity, Entity *child)
{
	BE_STATIC_PIMPL_HANDLE(entity);

	// Inform all controllers
	for (M::EntityControllers controllers = m.entities(M::controllers)[entity.Index];
		controllers; ++controllers)
		m.controllerPool[controllers.Begin]->ChildRemoved(child);
}

// Sets the owner of this entity.
void Entities::SetOwner(EntityHandle entity, EntityController *pOwner, EntityOwnerNotification::T notification)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	M::Registry &reg = m.entities[entity.Index];

	if (pOwner != reg.pOwner)
	{
		Entity *childEntity = m.entities(M::reflected)[entity.Index];

		if (reg.pOwner && reg.pOwner->ChildRemoved(childEntity))
			reg.pOwner = nullptr;
		
		if (pOwner && !reg.pOwner)
			if (notification == EntityOwnerNotification::WithoutNotification || pOwner->ChildAdded(childEntity))
				reg.pOwner = pOwner;
	}
}

// The OWNER releases the entity back into the wild.
void Entities::UnsetOwner(EntityHandle entity, EntityController *owner, EntityOwnerNotification::T notification)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	M::Registry &reg = m.entities[entity.Index];

	// TODO: ASSERTEXCEPT: Release assert -> exception?
	LEAN_ASSERT_DEBUG(owner && reg.pOwner == owner);

	if (owner && reg.pOwner == owner)
	{
		Entity *childEntity = m.entities(M::reflected)[entity.Index];

		if (notification == EntityOwnerNotification::WithoutNotification || owner->ChildRemoved(childEntity))
			reg.pOwner = nullptr;
	}
}

// Gets the owner of this entity.
EntityController* Entities::GetOwner(const EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE_CONST(entity);
	return m.entities[entity.Index].pOwner;
}

// Attaches the entity.
void Entities::Attach(EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE(entity);

	M::State &state = m.entities(M::state)[entity.Index];

	if (!state.Attached)
	{
		Entity *handle = m.entities(M::reflected)[entity.Index];

		// Attach all controllers
		for (M::EntityControllers controllers = m.entities(M::controllers)[entity.Index];
			controllers.Begin < controllers.End; ++controllers.Begin)
			m.controllerPool[controllers.Begin]->Attach(handle);

		state.Attached = true;
		ScheduleSync(m, entity.Index);
	}
}
// Detaches the entity.
void Entities::Detach(EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE(entity);

	M::State &state = m.entities(M::state)[entity.Index];

	if (state.Attached)
	{
		Entity *handle = m.entities(M::reflected)[entity.Index];

		// Detach controllers in reverse order
		for (M::EntityControllers controllers = m.entities(M::controllers)[entity.Index];
			controllers.End-- > controllers.Begin; )
			m.controllerPool[controllers.End]->Detach(handle);

		state.Attached = false;
	}
}
// Checks whether the given entity is attached.
bool Entities::IsAttached(const EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE_CONST(entity);
	return m.entities(M::state)[entity.Index].Attached;
}

// Sets the name.
void Entities::SetName(EntityHandle entity, const utf8_ntri &name)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	m.entities(M::registry)[entity.Index].Name = name.to<utf8_string>();
	PropertyChanged(m, entity.Index);
}
// Gets the name.
const utf8_string& Entities::GetName(const EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE_CONST(entity);
	return m.entities(M::registry)[entity.Index].Name;
}

// Sets the persistent ID.
void Entities::SetPersistentID(EntityHandle entity, uint8 persistentID)
{
	BE_STATIC_PIMPL_HANDLE(entity);
	M::Registry &entityReg = m.entities[entity.Index];

	if (persistentID == NewPersistentID)
		persistentID = m.persistentIDs->ReserveID();

	if (persistentID != entityReg.PersistentID)
	{
		Entity *handle = m.entities(M::reflected)[entity.Index];

		// Re-register persistent entity
		if (persistentID == AnonymousPersistentID ||
			m.persistentIDs->SetReference(persistentID, handle, true))
		{
			uint8 oldPersistentID = entityReg.PersistentID;
			entityReg.PersistentID = persistentID;
			m.persistentIDs->UnsetReference(oldPersistentID, handle);
		}
		else
			LEAN_LOG_ERROR_CTX("Persistent entity ID collision", entityReg.Name.c_str());
	}
}
// Gets the persistent ID.
uint8 Entities::GetPersistentID(const EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE_CONST(entity);
	return m.entities(M::registry)[entity.Index].PersistentID;
}

// Sets the custom ID.
void Entities::SetCustomIDBase(uint4 baseID)
{
	LEAN_STATIC_PIMPL();
	m.customBaseID = baseID;
}

/// Gets an entity from the given custom ID.
Entity* Entities::GetEntityByCustomID(uint4 customID)
{
	LEAN_STATIC_PIMPL();
	uint4 internalIdx = customID - m.customBaseID;
	return (internalIdx < m.entities.size())
		? m.entities(M::reflected)[internalIdx]
		: nullptr;
}

// Gets the custom ID.
uint4 Entities::GetCustomID(const EntityHandle entity)
{
	BE_STATIC_PIMPL_HANDLE_CONST(entity);
	return m.customBaseID + entity.Index;
}

// Adds a property listener.
void Entity::AddObserver(beCore::ComponentObserver *listener)
{
	BE_FREE_STATIC_PIMPL_HANDLE(Entities, m_handle);
	m.entities(M::observers)[m_handle.Index].AddObserver(listener);
}

// Removes a property listener.
void Entity::RemoveObserver(beCore::ComponentObserver *pListener)
{
	BE_FREE_STATIC_PIMPL_HANDLE(Entities, m_handle);
	m.entities(M::observers)[m_handle.Index].RemoveObserver(pListener);
}

// Hints at externally imposed changes, such as changes via an editor UI.
void Entity::ForcedChangeHint()
{
	NeedSync();
}

// Gets the first accessible entity (candidate or parent) for the given entity, null if none.
Entity* FirstAccessibleEntity(Entity *pCandidate)
{
	for (bool foundAccessible = false; pCandidate && !foundAccessible; )
	{
		foundAccessible = true;

		if (bees::EntityController *owner = pCandidate->GetOwner())
		{
			foundAccessible = (owner->GetChildFlags() & bees::ChildEntityFlags::Accessible) != 0;

			if (!foundAccessible)
				pCandidate = owner->GetParent();
		}
	}

	return pCandidate;
}

// Gets the next accessible parent entity for the given entity, null if none.
Entity* NextAccessibleEntity(Entity *pChild)
{
	if (pChild)
	{
		if (bees::EntityController *owner = pChild->GetOwner())
			return FirstAccessibleEntity(owner->GetParent());
	}
	return nullptr;
}

// Attaches the given collection of entities.
void Attach(Entity *const *entities, uint4 count)
{
	Entity *const *attached = entities, *const *end = entities + count;

	try
	{
		for (; attached < end; ++attached)
			(*attached)->Attach();
	}
	catch (...)
	{
		Detach(entities, (uint4) (attached - entities));
		throw;
	}
}

// Detaches the given collection of entities.
void Detach(Entity *const *entities, uint4 count)
{
	for (Entity *const *it = entities + count; it-- > entities; )
		(*it)->Detach();
}

} // namespace
