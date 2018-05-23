/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERABLE_PROCESSING_EFFECT_DRIVER
#define BE_SCENE_RENDERABLE_PROCESSING_EFFECT_DRIVER

#include "beScene.h"
#include "beRenderableEffectDriver.h"
#include "bePipeEffectBinder.h"
#include <beGraphics/beEffect.h>
#include <beGraphics/beDeviceContext.h>
#include <beGraphics/beStateManager.h>

namespace beScene
{

/// Renderable effect binder.
class RenderableProcessingEffectDriver : public RenderableEffectDriver
{
protected:
	PipeEffectBinder m_pipeBinder;		///< Pipe effect binder.

public:
	/// Constructor.
	BE_SCENE_API RenderableProcessingEffectDriver(const beGraphics::Technique &technique, RenderingPipeline *pPipeline, PerspectiveEffectBinderPool *pPool,
		uint4 flags = 0);
	/// Destructor.
	BE_SCENE_API ~RenderableProcessingEffectDriver();

	/// Draws the given pass.
	BE_SCENE_API void Render(const QueuedPass *pPass, const RenderableEffectData *pRenderableData, const Perspective &perspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context) const LEAN_OVERRIDE;

	/// Gets the pipe effect binder.
	LEAN_INLINE const PipeEffectBinder& GetPipeBinder() const { return m_pipeBinder; }
};

} // namespace

#endif