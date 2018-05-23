/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_LIGHT_EFFECT_DRIVER
#define BE_SCENE_LIGHT_EFFECT_DRIVER

#include "beScene.h"
#include "beAbstractLightEffectDriver.h"
#include "bePipelineEffectBinder.h"
#include "bePerspectiveEffectBinder.h"
#include "beLightEffectBinder.h"
#include "bePipeEffectBinder.h"
#include <beGraphics/beEffect.h>
#include <beGraphics/beDeviceContext.h>
#include <beGraphics/beStateManager.h>

namespace beScene
{

/// Light effect driver state.
struct LightDriverState
{
	uint4 PassID;				///< Pass id.

	LightDriverState()
		: PassID() { }
};

/// Casts the given abstract state data into renderable driver state.
template <class LightDriverState>
LEAN_INLINE LightDriverState& ToLightDriverState(AbstractLightDriverState &state)
{
	LEAN_STATIC_ASSERT(sizeof(LightDriverState) <= sizeof(AbstractLightDriverState));
	return *reinterpret_cast<LightDriverState*>(state.Data);
}

/// Light effect driver.
class LightEffectDriver : public AbstractLightEffectDriver
{
protected:
	PipelineEffectBinder m_pipelineBinder;			///< Pipeline effect binder.
	PerspectiveEffectBinder m_perspectiveBinder;	///< Renderable effect binder.
	LightEffectBinder m_lightBinder;				///< Light effect binder.
	PipeEffectBinder m_pipeBinder;					///< Pipe effect binder.

public:
	/// Constructor.
	BE_SCENE_API LightEffectDriver(const beGraphics::Technique &technique, RenderingPipeline *pipeline, PerspectiveEffectBinderPool *pool,
		uint4 flags = 0);
	/// Destructor.
	BE_SCENE_API ~LightEffectDriver();

	/// Draws the given pass.
	BE_SCENE_API void Render(const QueuedPass *pass, const LightEffectData *data, const Perspective &perspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context) const LEAN_OVERRIDE;

	/// Gets the passes.
	BE_SCENE_API PassRange GetPasses() const LEAN_OVERRIDE;

	/// Gets the pipeline effect binder.
	LEAN_INLINE const PipelineEffectBinder& GetPipelineBinder() const { return m_pipelineBinder; }
	/// Gets the perspective effect binder.
	LEAN_INLINE const PerspectiveEffectBinder& GetPerspectiveBinder() const { return m_perspectiveBinder; }

	/// Gets the effect.
	LEAN_INLINE const beGraphics::Effect& GetEffect() const { return m_pipelineBinder.GetEffect(); }
};

} // namespace

#endif