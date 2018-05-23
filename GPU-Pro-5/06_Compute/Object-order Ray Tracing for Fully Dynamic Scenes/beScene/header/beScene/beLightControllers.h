/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_LIGHT_CONTROLLERS
#define BE_SCENE_LIGHT_CONTROLLERS

#include "beScene.h"
#include <lean/pimpl/static_pimpl.h>
#include <beCore/beShared.h>
#include <beCore/beMany.h>
#include <beCore/beComponentMonitor.h>
#include "beRenderable.h"
#include "beRenderingLimits.h"
#include <beEntitySystem/beEntityController.h>
#include <beEntitySystem/beSimulationController.h>
#include <beEntitySystem/beRenderable.h>
#include <beEntitySystem/beSynchronized.h>
#include <beGraphics/beDevice.h>
#include <lean/smart/scoped_ptr.h>

#include <lean/smart/resource_ptr.h>
#include <lean/containers/dynamic_array.h>

#include "beMath/beSphereDef.h"

namespace beScene
{

// Prototypes
class AssembledMesh;
class LightMaterial;
class PerspectivePool;
class RenderContext;
class RenderingPipeline;

class ResourceManager;
class EffectDrivenRenderer;

class DirectionalLightController;
class PointLightController;
class SpotLightController;

template <class LightController>
class LightControllers;
template <class LightController>
class LightControllerBase;

/// Handle to a mesh controller.
template <class LightController>
struct LightControllerHandle : public beCore::GroupElementHandle< LightControllers<LightController> >
{
	friend LightControllers<LightController>;

private:
	/// Internal constructor.
	LightControllerHandle(LightControllers<LightController> *controllers, uint4 internalID)
		: GroupElementHandle< LightControllers<LightController> >(controllers, internalID) { }
};

class LEAN_INTERFACE LightControllersBase : public beEntitySystem::WorldController,
	public beEntitySystem::Synchronized, public Renderable
{
	LEAN_INTERFACE_BEHAVIOR(LightControllersBase)

public:
	struct M;
};

template <class LightController>
struct LightControllersColorBase
{
	typedef LightController LightController;
	typedef LightControllers<LightController> LightControllers;
	typedef LightControllerHandle<LightController> LightControllerHandle;

	/// Sets the color.
	BE_SCENE_API static void SetColor(LightControllerHandle controller, const beMath::fvec4 &color);
	/// Gets the color.
	BE_SCENE_API static const beMath::fvec4& GetColor(const LightControllerHandle controller);

	/// Sets the (indirect) color.
	BE_SCENE_API static void SetIndirectColor(LightControllerHandle controller, const beMath::fvec4 &color);
	/// Gets the (indirect) color.
	BE_SCENE_API static const beMath::fvec4& GetIndirectColor(const LightControllerHandle controller);
};

template <class LightController>
struct LightControllersPointBase : LightControllersColorBase<LightController>
{
	typedef LightController LightController;
	typedef typename LightControllersColorBase<LightController>::LightControllers LightControllers;
	typedef typename LightControllersColorBase<LightController>::LightControllerHandle LightControllerHandle;

	/// Sets the attenuation.
	BE_SCENE_API static void SetAttenuation(LightControllerHandle controller, float attenuation);
	/// Gets the attenuation.
	BE_SCENE_API static float GetAttenuation(const LightControllerHandle controller);

	/// Sets the attenuation offset.
	BE_SCENE_API static void SetAttenuationOffset(LightControllerHandle controller, float offset);
	/// Gets the attenuation offset.
	BE_SCENE_API static float GetAttenuationOffset(const LightControllerHandle controller);

	/// Sets the range.
	BE_SCENE_API static void SetRange(LightControllerHandle controller, float range);
	/// Gets the range.
	BE_SCENE_API static float GetRange(LightControllerHandle controller);
};

template <class LightController>
struct LightControllersSpotBase : LightControllersPointBase<LightController>
{
	typedef LightController LightController;
	typedef typename LightControllersSpotBase<LightController>::LightControllers LightControllers;
	typedef typename LightControllersPointBase<LightController>::LightControllerHandle LightControllerHandle;

	/// Sets the angles.
	BE_SCENE_API static void SetInnerAngle(LightControllerHandle controller, float angle);
	/// Gets the angles.
	BE_SCENE_API static float GetInnerAngle(const LightControllerHandle controller);

	/// Sets the angles.
	BE_SCENE_API static void SetOuterAngle(LightControllerHandle controller, float angle);
	/// Gets the angles.
	BE_SCENE_API static float GetOuterAngle(const LightControllerHandle controller);
};

template <class LightController>
struct LightControllersTypedBase;
template<> struct LightControllersTypedBase<DirectionalLightController> : LightControllersColorBase<DirectionalLightController> { };
template<> struct LightControllersTypedBase<PointLightController> : LightControllersPointBase<PointLightController> { };
template<> struct LightControllersTypedBase<SpotLightController> : LightControllersSpotBase<SpotLightController> { };

/// Light controller manager.
template <class LightController>
class LEAN_INTERFACE LightControllers : public LightControllersTypedBase<LightController>, public LightControllersBase
{
	LEAN_SHARED_SIMPL_INTERFACE_BEHAVIOR(LightControllers)

public:
	class M;

	typedef LightController LightController;
	typedef typename LightControllersTypedBase<LightController>::LightControllerHandle LightControllerHandle;

	/// Adds a mesh controller.
	BE_SCENE_API LightController* AddController();
	/// Clones the given controller.
	BE_SCENE_API static LightController* CloneController(const LightControllerHandle controller);
	/// Removes a mesh controller.
	BE_SCENE_API static void RemoveController(LightController *pController);

	/// Commits changes.
	BE_SCENE_API void Commit();
	
	/// Perform visiblity culling.
	BE_SCENE_API void Cull(PipelinePerspective &perspective) const LEAN_OVERRIDE;
	/// Prepares the given render queue for the given perspective, returning true if active.
	BE_SCENE_API bool Prepare(PipelinePerspective &perspective, PipelineQueueID queueID,
		const PipelineStageDesc &stageDesc, const RenderQueueDesc &queueDesc) const LEAN_OVERRIDE;
	/// Prepares the collected render queues for the given perspective.
	BE_SCENE_API void Collect(PipelinePerspective &perspective) const LEAN_OVERRIDE;
	/// Performs optional optimization such as sorting.
	BE_SCENE_API void Optimize(const PipelinePerspective &perspective, PipelineQueueID queueID) const LEAN_OVERRIDE;
	/// Prepares rendering from the collected render queues for the given perspective.
	BE_SCENE_API void PreRender(const PipelinePerspective &perspective, const RenderContext &context) const LEAN_OVERRIDE;
	/// Renders the given render queue for the given perspective.
	BE_SCENE_API void Render(const PipelinePerspective &perspective, PipelineQueueID queueID, const RenderContext &context) const LEAN_OVERRIDE;
	/// Renders the given single object for the given perspective.
	BE_SCENE_API void Render(uint4 objectID, const PipelinePerspective &perspective, PipelineQueueID queueID, const RenderContext &context) const LEAN_OVERRIDE;

	/// Attaches the controller to the given entity.
	BE_SCENE_API static void Attach(LightControllerHandle controller, beEntitySystem::Entity *entity);
	/// Detaches the controller from the given entity.
	BE_SCENE_API static void Detach(LightControllerHandle controller, beEntitySystem::Entity *entity);

	/// Sets the material.
	BE_SCENE_API static void SetMaterial(LightControllerHandle controller, LightMaterial *pMaterial);
	/// Gets the material.
	BE_SCENE_API static LightMaterial* GetMaterial(const LightControllerHandle controller);

	/// Enables shadow casting.
	BE_SCENE_API static void EnableShadow(LightControllerHandle controller, bool bEnable);
	/// Checks if this light is currently casting shadows.
	BE_SCENE_API static bool IsShadowEnabled(const LightControllerHandle controller);

	/// Sets the shadow stage mask.
	BE_SCENE_API static void SetShadowStages(LightControllerHandle controller, PipelineStageMask shadowStages);
	/// Gets the shadow stage mask.
	BE_SCENE_API static PipelineStageMask GetShadowStages(const LightControllerHandle controller);

	/// Sets the shadow resolution.
	BE_SCENE_API static void SetShadowResolution(LightControllerHandle controller, uint4 resolution);
	/// Gets the shadow resolution.
	BE_SCENE_API static uint4 GetShadowResolution(const LightControllerHandle controller);

	/// Sets the visibility.
	BE_SCENE_API static void SetVisible(LightControllerHandle controller, bool bVisible);
	/// Gets the visibility.
	BE_SCENE_API static bool IsVisible(const LightControllerHandle controller);
	
	/// Sets the local bounding sphere.
	BE_SCENE_API static void SetLocalBounds(LightControllerHandle controller, const beMath::fsphere3 &bounds);
	/// Gets the local bounding sphere.
	BE_SCENE_API static const beMath::fsphere3& GetLocalBounds(const LightControllerHandle controller);
	
	/// Sets the component monitor.
	BE_SCENE_API void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor);
	/// Gets the component monitor.
	BE_SCENE_API beCore::ComponentMonitor* GetComponentMonitor() const;

	/// Gets the controller type.
	BE_SCENE_API static const beCore::ComponentType* GetComponentType();
	/// Gets the controller type.
	BE_SCENE_API const beCore::ComponentType* GetType() const;
};

/// Light controller.
template <class LightController>
class LightControllerBase : public lean::noncopyable, public beEntitySystem::EntityController
{
public:
	typedef LightControllers<LightController> LightControllers;
	typedef typename LightControllers::LightControllerHandle LightControllerHandle;

	friend LightControllers;

protected:
	LightControllerHandle m_handle;

	/// Internal constructor.
	LightControllerBase(LightControllerHandle handle)
		: m_handle(handle) { }

public:
	/// Synchronizes this controller with the given entity controlled.
	BE_SCENE_API void Flush(const beEntitySystem::EntityHandle entity);

	/// Sets the material.
	LEAN_INLINE void SetMaterial(LightMaterial *pMaterial) { LightControllers::SetMaterial(m_handle, pMaterial); }
	/// Gets the material.
	LEAN_INLINE LightMaterial* GetMaterial() const { return LightControllers::GetMaterial(m_handle); }

	/// Enables shadow casting.
	LEAN_INLINE void EnableShadow(bool bEnable) { LightControllers::EnableShadow(m_handle, bEnable); }
	/// Checks if this light is currently casting shadows.
	LEAN_INLINE bool IsShadowEnabled() const { return LightControllers::IsShadowEnabled(m_handle); }

	/// Sets the shadow stage mask.
	LEAN_INLINE void SetShadowStages(PipelineStageMask shadowStages) { LightControllers::SetShadowStages(m_handle, shadowStages); }
	/// Gets the shadow stage mask.
	LEAN_INLINE PipelineStageMask GetShadowStages() const { return LightControllers::GetShadowStages(m_handle); }

	/// Sets the shadow resolution.
	LEAN_INLINE void SetShadowResolution(uint4 resolution) { LightControllers::SetShadowResolution(m_handle, resolution); }
	/// Gets the shadow resolution.
	LEAN_INLINE uint4 GetShadowResolution() const { return LightControllers::GetShadowResolution(m_handle); }

	/// Sets the visibility.
	LEAN_INLINE void SetVisible(bool bVisible) { LightControllers::SetVisible(m_handle, bVisible); }
	/// Gets the visibility.
	LEAN_INLINE bool IsVisible() const { return LightControllers::IsVisible(m_handle); }

	/// Sets the local bounding sphere.
	LEAN_INLINE void SetLocalBounds(const beMath::fsphere3 &bounds) { LightControllers::SetLocalBounds(m_handle, bounds); }
	/// Gets the local bounding sphere.
	LEAN_INLINE const beMath::fsphere3& GetLocalBounds() { return LightControllers::GetLocalBounds(m_handle); }

	/// Attaches the entity.
	LEAN_INLINE void Attach(beEntitySystem::Entity *entity) { LightControllers::Attach(m_handle, entity); }
	/// Detaches the entity.
	LEAN_INLINE void Detach(beEntitySystem::Entity *entity) { LightControllers::Detach(m_handle, entity); }
	
	/// Adds a property listener.
	BE_SCENE_API void AddObserver(beCore::ComponentObserver *listener) LEAN_OVERRIDE;
	/// Removes a property listener.
	BE_SCENE_API void RemoveObserver(beCore::ComponentObserver *pListener) LEAN_OVERRIDE;

	/// Gets the reflection properties.
	BE_SCENE_API static Properties GetOwnProperties();
	/// Gets the reflection properties.
	BE_SCENE_API Properties GetReflectionProperties() const;

	/// Gets the number of child components.
	BE_SCENE_API uint4 GetComponentCount() const;
	/// Gets the name of the n-th child component.
	BE_SCENE_API beCore::Exchange::utf8_string GetComponentName(uint4 idx) const;
	/// Gets the n-th reflected child component, nullptr if not reflected.
	BE_SCENE_API lean::com_ptr<const ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx) const;

	/// Gets the type of the n-th child component.
	BE_SCENE_API const beCore::ComponentType* GetComponentType(uint4 idx) const;
	/// Gets the n-th component.
	BE_SCENE_API lean::cloneable_obj<lean::any, true> GetComponent(uint4 idx) const;
	/// Returns true, if the n-th component can be replaced.
	BE_SCENE_API bool IsComponentReplaceable(uint4 idx) const;
	/// Sets the n-th component.
	BE_SCENE_API void SetComponent(uint4 idx, const lean::any &pComponent);

	/// Gets the controller type.
	BE_SCENE_API static const beCore::ComponentType* GetComponentType();
	/// Gets the controller type.
	BE_SCENE_API const beCore::ComponentType* GetType() const;
	
	/// Clones this entity controller.
	BE_SCENE_API LightControllerBase* Clone() const { return LightControllers::CloneController(m_handle); }
	/// Removes this controller.
	BE_SCENE_API void Abandon() const { LightControllers::RemoveController((LightController*) this); }

	/// Gets the handle to the entity.
	LEAN_INLINE LightControllerHandle& Handle() { return m_handle; }
	/// Gets the handle to the entity.
	LEAN_INLINE const LightControllerHandle& Handle() const { return m_handle; }
};

/// Light controller.
class DirectionalLightController : public LightControllerBase<DirectionalLightController>
{
public:
	typedef LightControllerBase<DirectionalLightController>::LightControllers LightControllers;
	typedef LightControllers::LightControllerHandle LightControllerHandle;

	friend LightControllers;

private:
	/// Internal constructor.
	DirectionalLightController(LightControllerHandle handle)
		: LightControllerBase<DirectionalLightController>(handle) { }

public:
	/// Sets the color.
	LEAN_INLINE void SetColor(const beMath::fvec4 &color) { LightControllers::SetColor(m_handle, color); }
	/// Gets the color.
	LEAN_INLINE const beMath::fvec4& GetColor() const { return LightControllers::GetColor(m_handle); }

	/// Sets the color.
	LEAN_INLINE void SetIndirectColor(const beMath::fvec4 &color) { LightControllers::SetIndirectColor(m_handle, color); }
	/// Gets the color.
	LEAN_INLINE const beMath::fvec4& GetIndirectColor() const { return LightControllers::GetIndirectColor(m_handle); }
};

/// Light controller.
class PointLightController : public LightControllerBase<PointLightController>
{
public:
	typedef LightControllerBase<PointLightController>::LightControllers LightControllers;
	typedef LightControllers::LightControllerHandle LightControllerHandle;

	friend LightControllers;

private:
	/// Internal constructor.
	PointLightController(LightControllerHandle handle)
		: LightControllerBase<PointLightController>(handle) { }

public:
	/// Sets the color.
	LEAN_INLINE void SetColor(const beMath::fvec4 &color) { LightControllers::SetColor(m_handle, color); }
	/// Gets the color.
	LEAN_INLINE const beMath::fvec4& GetColor() const { return LightControllers::GetColor(m_handle); }

	/// Sets the color.
	LEAN_INLINE void SetIndirectColor(const beMath::fvec4 &color) { LightControllers::SetIndirectColor(m_handle, color); }
	/// Gets the color.
	LEAN_INLINE const beMath::fvec4& GetIndirectColor() const { return LightControllers::GetIndirectColor(m_handle); }

	/// Sets the attenuation.
	LEAN_INLINE void SetAttenuation(float attenuation) { LightControllers::SetAttenuation(m_handle, attenuation); }
	/// Gets the attenuation.
	LEAN_INLINE float GetAttenuation() const { return LightControllers::GetAttenuation(m_handle); }

	/// Sets the attenuation offset.
	LEAN_INLINE void SetAttenuationOffset(float attenuationOffset) { LightControllers::SetAttenuationOffset(m_handle, attenuationOffset); }
	/// Gets the attenuation offset.
	LEAN_INLINE float GetAttenuationOffset() const { return LightControllers::GetAttenuationOffset(m_handle); }
	
	/// Sets the range.
	LEAN_INLINE void SetRange(float range) { LightControllers::SetRange(m_handle, range); }
	/// Gets the range.
	LEAN_INLINE float GetRange() const { return LightControllers::GetRange(m_handle); }
};

/// Light controller.
class SpotLightController : public LightControllerBase<SpotLightController>
{
public:
	typedef LightControllerBase<SpotLightController>::LightControllers LightControllers;
	typedef LightControllers::LightControllerHandle LightControllerHandle;

	friend LightControllers;

private:
	/// Internal constructor.
	SpotLightController(LightControllerHandle handle)
		: LightControllerBase<SpotLightController>(handle) { }

public:
	/// Sets the color.
	LEAN_INLINE void SetColor(const beMath::fvec4 &color) { LightControllers::SetColor(m_handle, color); }
	/// Gets the color.
	LEAN_INLINE const beMath::fvec4& GetColor() const { return LightControllers::GetColor(m_handle); }

	/// Sets the color.
	LEAN_INLINE void SetIndirectColor(const beMath::fvec4 &color) { LightControllers::SetIndirectColor(m_handle, color); }
	/// Gets the color.
	LEAN_INLINE const beMath::fvec4& GetIndirectColor() const { return LightControllers::GetIndirectColor(m_handle); }

	/// Sets the attenuation.
	LEAN_INLINE void SetAttenuation(float attenuation) { LightControllers::SetAttenuation(m_handle, attenuation); }
	/// Gets the attenuation.
	LEAN_INLINE float GetAttenuation() const { return LightControllers::GetAttenuation(m_handle); }

	/// Sets the attenuation offset.
	LEAN_INLINE void SetAttenuationOffset(float attenuationOffset) { LightControllers::SetAttenuationOffset(m_handle, attenuationOffset); }
	/// Gets the attenuation offset.
	LEAN_INLINE float GetAttenuationOffset() const { return LightControllers::GetAttenuationOffset(m_handle); }
	
	/// Sets the range.
	LEAN_INLINE void SetRange(float range) { LightControllers::SetRange(m_handle, range); }
	/// Gets the range.
	LEAN_INLINE float GetRange() const { return LightControllers::GetRange(m_handle); }
	
	/// Sets the angles.
	LEAN_INLINE void SetInnerAngle(float innerAngle) { LightControllers::SetInnerAngle(m_handle, innerAngle); }
	/// Gets the inner angle.
	LEAN_INLINE float GetInnerAngle() const { return LightControllers::GetInnerAngle(m_handle); }

	/// Sets the angles.
	LEAN_INLINE void SetOuterAngle(float outerAngle) { LightControllers::SetOuterAngle(m_handle, outerAngle); }
	/// Gets the outer angle.
	LEAN_INLINE float GetOuterAngle() const { return LightControllers::GetOuterAngle(m_handle); }
};

/// Creates a collection of light controllers.
/// @relatesalso LightControllers
template <class LightController>
BE_SCENE_API lean::scoped_ptr<LightControllers<LightController>, lean::critical_ref> CreateLightControllers(beCore::PersistentIDs *persistentIDs,
	PerspectivePool *perspectivePool, const RenderingPipeline &pipeline, const beGraphics::Device &device);

/// Sets the default shadow stage for directional lights.
template <class LightController>
BE_SCENE_API void SetDefaultShadowStage(const utf8_ntri &name);
/// Gets the default shadow stage for directional lights.
template <class LightController>
BE_SCENE_API utf8_ntr GetDefaultShadowStage();
/// Gets the default shadow stage for directional lights.
template <class LightController>
BE_SCENE_API PipelineStageMask GetDefaultShadowStage(const RenderingPipeline &pipeline);

/// Sets the default light effect file.
template <class LightController>
BE_SCENE_API void SetLightDefaultEffect(const utf8_ntri &file);
/// Gets the default mesh effect file.
template <class LightController>
BE_SCENE_API utf8_ntr GetLightDefaultEffect();
/// Gets the default material for meshes.
template <class LightController>
BE_SCENE_API LightMaterial* GetLightDefaultMaterial(ResourceManager &resources, EffectDrivenRenderer &renderer);

template <class LightController>
void InstantiateLightControllerFunctions()
{
	lean::absorb
	(
		(lean::absorbfun) &CreateLightControllers<LightController>,

		(lean::absorbfun) &SetDefaultShadowStage<LightController>,
		(lean::absorbfun) (utf8_ntr (*)()) &GetDefaultShadowStage<LightController>,
		(lean::absorbfun) (PipelineStageMask (*)(const RenderingPipeline&)) &GetDefaultShadowStage<LightController>,

		(lean::absorbfun) &SetLightDefaultEffect<LightController>,
		(lean::absorbfun) &GetLightDefaultEffect<LightController>,
		(lean::absorbfun) &GetLightDefaultMaterial<LightController>
	);
}

} // namespace

#endif