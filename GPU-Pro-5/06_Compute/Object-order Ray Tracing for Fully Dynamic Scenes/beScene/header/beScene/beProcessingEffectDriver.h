/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PROCESSING_EFFECT_DRIVER
#define BE_SCENE_PROCESSING_EFFECT_DRIVER

#include "beScene.h"
#include "beAbstractProcessingEffectDriver.h"
#include "bePipelineEffectBinder.h"
#include "bePerspectiveEffectBinder.h"
#include "bePipeEffectBinder.h"
#include <beGraphics/beEffect.h>
#include <beGraphics/beDeviceContext.h>
#include <beGraphics/beStateManager.h>

namespace beScene
{

/// Processing effect binder.
class ProcessingEffectDriver : public AbstractProcessingEffectDriver
{
protected:
	/// Pipeline effect binder.
	PipelineEffectBinder m_pipelineBinder;
	/// Perspective effect binder.
	PerspectiveEffectBinder m_perspectiveBinder;
	/// Pipe effect binder.
	PipeEffectBinder m_pipeBinder;

public:
	/// Constructor.
	BE_SCENE_API ProcessingEffectDriver(const beGraphics::Technique &technique, RenderingPipeline *pPipeline, PerspectiveEffectBinderPool *pPool);
	/// Destructor.
	BE_SCENE_API ~ProcessingEffectDriver();

	/// Draws the given pass.
	BE_SCENE_API void Render(const QueuedPass *pPass, const void *pProcessor, const Perspective *pPerspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context) const;

	/// Gets the passes.
	BE_SCENE_API PassRange GetPasses() const;

	/// Gets the pipeline effect binder.
	LEAN_INLINE const PipelineEffectBinder& GetPipelineBinder() const { return m_pipelineBinder; }
	/// Gets the perspective effect binder.
	LEAN_INLINE const PerspectiveEffectBinder& GetPerspectiveBinder() const { return m_perspectiveBinder; }
	/// Gets the pipe effect binder.
	LEAN_INLINE const PipeEffectBinder& GetPipeBinder() const { return m_pipeBinder; }

	/// Gets the effect.
	LEAN_INLINE const beGraphics::Effect& GetEffect() const { return m_perspectiveBinder.GetEffect(); }
};

} // namespace

#endif