/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERABLE_EFFECT_DRIVER
#define BE_SCENE_RENDERABLE_EFFECT_DRIVER

#include "beScene.h"
#include "beAbstractRenderableEffectDriver.h"
#include "bePipelineEffectBinder.h"
#include "beRenderableEffectBinder.h"
#include "beLightingEffectBinder.h"
#include <beGraphics/beEffect.h>
#include <beGraphics/beDeviceContext.h>
#include <beGraphics/beStateManager.h>

namespace beScene
{

/// Renderable effect driver state.
struct RenderableDriverState
{
	LightingBinderState Light;	///< Light binder state.
	uint4 PassID;				///< Pass id.

	RenderableDriverState()
		: PassID() { }
};

/// Casts the given abstract state data into renderable driver state.
template <class RenderableDriverState>
LEAN_INLINE RenderableDriverState& ToRenderableDriverState(AbstractRenderableDriverState &state)
{
	LEAN_STATIC_ASSERT(sizeof(RenderableDriverState) <= sizeof(AbstractRenderableDriverState));
	return *reinterpret_cast<RenderableDriverState*>(state.Data);
}

/// Renderable effect driver.
class RenderableEffectDriver : public AbstractRenderableEffectDriver
{
protected:
	PipelineEffectBinder m_pipelineBinder;		///< Pipeline effect binder.
	RenderableEffectBinder m_renderableBinder;	///< Renderable effect binder.
	LightingEffectBinder m_lightBinder;			///< Light effect binder.

public:
	/// Constructor.
	BE_SCENE_API RenderableEffectDriver(const beGraphics::Technique &technique, RenderingPipeline *pPipeline, PerspectiveEffectBinderPool *pPool,
		uint4 flags = 0);
	/// Destructor.
	BE_SCENE_API ~RenderableEffectDriver();

	/// Draws the given pass.
	BE_SCENE_API void Render(const QueuedPass *pPass, const RenderableEffectData *pRenderableData, const Perspective &perspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context) const LEAN_OVERRIDE;

	/// Gets the passes.
	BE_SCENE_API PassRange GetPasses() const LEAN_OVERRIDE;

	/// Gets the pipeline effect binder.
	LEAN_INLINE const PipelineEffectBinder& GetPipelineBinder() const { return m_pipelineBinder; }
	/// Gets the renderable effect binder.
	LEAN_INLINE const RenderableEffectBinder& GetRenderableBinder() const { return m_renderableBinder; }

	/// Gets the effect.
	const beGraphics::Effect& GetEffect() const LEAN_OVERRIDE { return m_pipelineBinder.GetEffect(); }
};

} // namespace

#endif