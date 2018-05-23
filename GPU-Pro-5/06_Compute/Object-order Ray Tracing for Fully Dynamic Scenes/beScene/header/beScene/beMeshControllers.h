/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_MESH_CONTROLLERS
#define BE_SCENE_MESH_CONTROLLERS

#include "beScene.h"
#include <lean/pimpl/static_pimpl.h>
#include <beCore/beShared.h>
#include <beCore/beMany.h>
#include <beCore/beComponentMonitor.h>

#include <beEntitySystem/beEntityController.h>
#include <beEntitySystem/beSimulationController.h>
#include <beEntitySystem/beRenderable.h>
#include <beEntitySystem/beSynchronized.h>
#include "beRenderable.h"

#include <lean/smart/scoped_ptr.h>

#include "beMath/beSphereDef.h"

namespace beScene
{

// Prototypes
class RenderableMesh;
class RenderableMaterial;
class RenderContext;

// Prototypes
class MeshController;
class MeshControllers;

/// Handle to a mesh controller.
struct MeshControllerHandle : public beCore::GroupElementHandle<MeshControllers>
{
	friend MeshControllers;

private:
	/// Internal constructor.
	MeshControllerHandle(class MeshControllers *controllers, uint4 internalID)
		: GroupElementHandle<MeshControllers>(controllers, internalID) { }
};
/// Mesh controller manager.
class LEAN_INTERFACE MeshControllers : public beCore::Resource, public Renderable, 
	public beEntitySystem::WorldController
{
	LEAN_SHARED_SIMPL_INTERFACE_BEHAVIOR(MeshControllers)

public:
	class M;
	
	/// Adds a mesh controller.
	BE_SCENE_API MeshController* AddController();
	/// Clones the given controller.
	BE_SCENE_API static MeshController* CloneController(const MeshControllerHandle controller);
	/// Removes a mesh controller.
	BE_SCENE_API static void RemoveController(MeshController *pController);

	/// Commits changes.
	BE_SCENE_API void Commit();
	
/*	/// Flushes changes.
	BE_SCENE_API void Flush(MeshControllerHandle controller);
	/// Synchronizes this controller with the controlled entity.
	BE_SCENE_API void Synchronize(MeshControllerHandle controller);
*/
	/// Perform visiblity culling.
	BE_SCENE_API void Cull(PipelinePerspective &perspective) const LEAN_OVERRIDE;
	/// Prepares the given render queue for the given perspective, returning true if active.
	BE_SCENE_API bool Prepare(PipelinePerspective &perspective, PipelineQueueID queueID,
		const PipelineStageDesc &stageDesc, const RenderQueueDesc &queueDesc) const LEAN_OVERRIDE;
	/// Performs optional optimization such as sorting.
	BE_SCENE_API void Optimize(const PipelinePerspective &perspective, PipelineQueueID queueID) const LEAN_OVERRIDE;
	/// Renders the given render queue for the given perspective.
	BE_SCENE_API void Render(const PipelinePerspective &perspective, PipelineQueueID queueID, const RenderContext &context) const LEAN_OVERRIDE;
	/// Renders the given single object for the given perspective.
	BE_SCENE_API void Render(uint4 objectID, const PipelinePerspective &perspective, PipelineQueueID queueID, const RenderContext &context) const LEAN_OVERRIDE;

	/// Attaches the controller to the given entity.
	BE_SCENE_API static void Attach(MeshControllerHandle controller, beEntitySystem::Entity *entity);
	/// Detaches the controller from the given entity.
	BE_SCENE_API static void Detach(MeshControllerHandle controller, beEntitySystem::Entity *entity);

	/// Sets the mesh.
	BE_SCENE_API static void SetMesh(MeshControllerHandle controller, RenderableMesh *pMesh);
	/// Gets the mesh.
	BE_SCENE_API static RenderableMesh* GetMesh(const MeshControllerHandle controller);

	/// Sets the visibility.
	BE_SCENE_API static void SetVisible(MeshControllerHandle controller, bool bVisible);
	/// Gets the visibility.
	BE_SCENE_API static bool IsVisible(const MeshControllerHandle controller);

	/// Sets the local bounding sphere.
	BE_SCENE_API static void SetLocalBounds(MeshControllerHandle controller, const beMath::fsphere3 &bounds);
	/// Gets the local bounding sphere.
	BE_SCENE_API static const beMath::fsphere3& GetLocalBounds(const MeshControllerHandle controller);
	
	/// Sets the component monitor.
	BE_SCENE_API void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor);
	/// Gets the component monitor.
	BE_SCENE_API beCore::ComponentMonitor* GetComponentMonitor() const;

	/// Gets the controller type.
	BE_SCENE_API static const beCore::ComponentType* GetComponentType();
	/// Gets the controller type.
	BE_SCENE_API const beCore::ComponentType* GetType() const;
};

/// Mesh controller.
class MeshController : public lean::noncopyable, public beEntitySystem::EntityController
{
	friend class MeshControllers;

private:
	MeshControllerHandle m_handle;

	/// Internal constructor.
	MeshController(MeshControllerHandle handle)
		: m_handle(handle) { }

public:
	/// Synchronizes this controller with the given entity controlled.
	BE_SCENE_API void Flush(const beEntitySystem::EntityHandle entity) LEAN_OVERRIDE;

	/// Sets the mesh.
	LEAN_INLINE void SetMesh(RenderableMesh *pMesh) { MeshControllers::SetMesh(m_handle, pMesh); }
	/// Gets the mesh.
	LEAN_INLINE RenderableMesh* GetMesh() const { return MeshControllers::GetMesh(m_handle); }

	/// Sets the visibility.
	LEAN_INLINE void SetVisible(bool bVisible) { MeshControllers::SetVisible(m_handle, bVisible); }
	/// Gets the visibility.
	LEAN_INLINE bool IsVisible() const { return MeshControllers::IsVisible(m_handle); }

	/// Sets the local bounding sphere.
	LEAN_INLINE void SetLocalBounds(const beMath::fsphere3 &bounds) { MeshControllers::SetLocalBounds(m_handle, bounds); }
	/// Gets the local bounding sphere.
	LEAN_INLINE const beMath::fsphere3& GetLocalBounds() { return MeshControllers::GetLocalBounds(m_handle); }

	/// Attaches the entity.
	BE_SCENE_API void Attach(beEntitySystem::Entity *entity) LEAN_OVERRIDE { MeshControllers::Attach(m_handle, entity); }
	/// Detaches the entity.
	BE_SCENE_API void Detach(beEntitySystem::Entity *entity) LEAN_OVERRIDE { MeshControllers::Detach(m_handle, entity); }

	/// Gets the number of child components.
	BE_SCENE_API uint4 GetComponentCount() const LEAN_OVERRIDE;
	/// Gets the name of the n-th child component.
	BE_SCENE_API beCore::Exchange::utf8_string GetComponentName(uint4 idx) const LEAN_OVERRIDE;
	/// Gets the n-th reflected child component, nullptr if not reflected.
	BE_SCENE_API lean::com_ptr<const ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx) const LEAN_OVERRIDE;

	/// Gets the type of the n-th child component.
	BE_SCENE_API const beCore::ComponentType* GetComponentType(uint4 idx) const LEAN_OVERRIDE;
	/// Gets the n-th component.
	BE_SCENE_API lean::cloneable_obj<lean::any, true> GetComponent(uint4 idx) const LEAN_OVERRIDE;
	/// Returns true, if the n-th component can be replaced.
	BE_SCENE_API bool IsComponentReplaceable(uint4 idx) const LEAN_OVERRIDE;
	/// Sets the n-th component.
	BE_SCENE_API void SetComponent(uint4 idx, const lean::any &pComponent) LEAN_OVERRIDE;

	/// Gets the controller type.
	BE_SCENE_API static const beCore::ComponentType* GetComponentType();
	/// Gets the controller type.
	BE_SCENE_API const beCore::ComponentType* GetType() const LEAN_OVERRIDE;
	
	/// Clones this entity controller.
	BE_SCENE_API MeshController* Clone() const LEAN_OVERRIDE { return MeshControllers::CloneController(m_handle); }
	/// Removes this controller.
	BE_SCENE_API void Abandon() const LEAN_OVERRIDE { MeshControllers::RemoveController(const_cast<MeshController*>(this)); }

	/// Gets the handle to the entity.
	LEAN_INLINE MeshControllerHandle& Handle() { return m_handle; }
	/// Gets the handle to the entity.
	LEAN_INLINE const MeshControllerHandle& Handle() const { return m_handle; }
};

/// Creates a collection of mesh controllers.
/// @relatesalso MeshControllers
BE_SCENE_API lean::scoped_ptr<MeshControllers, lean::critical_ref> CreateMeshControllers(beCore::PersistentIDs *persistentIDs);

class ResourceManager;
class EffectDrivenRenderer;

/// Sets the default mesh effect file.
BE_SCENE_API void SetMeshDefaultEffect(const utf8_ntri &file);
/// Gets the default mesh effect file.
BE_SCENE_API beCore::Exchange::utf8_string GetMeshDefaultEffect();
/// Gets the default material for meshes.
BE_SCENE_API RenderableMaterial* GetMeshDefaultMaterial(ResourceManager &resources, EffectDrivenRenderer &renderer);

} // namespace

#endif