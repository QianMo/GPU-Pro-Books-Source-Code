/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_SHADER_DRIVEN_PIPELINE
#define BE_SCENE_SHADER_DRIVEN_PIPELINE

#include "beScene.h"
#include <beGraphics/beEffect.h>

namespace beScene
{

// Prototypes
class RenderingPipeline;
class AbstractRenderableEffectDriver;
template <class EffectBinder>
class EffectDriverCache;

/// Loads all pipeline stages from the given effect.
BE_SCENE_API void LoadPipelineStages(RenderingPipeline &pipeline, const beGraphics::Effect *effect,
	EffectDriverCache<AbstractRenderableEffectDriver> &effectCache);
/// Loads all render queues from the given effect.
BE_SCENE_API void LoadRenderQueues(RenderingPipeline &pipeline, const beGraphics::Effect *effect,
	EffectDriverCache<AbstractRenderableEffectDriver> &effectCache);

/// Loads all pipeline information from the given effect.
BE_SCENE_API void LoadRenderingPipeline(RenderingPipeline &pipeline, const beGraphics::Effect *effect,
	EffectDriverCache<AbstractRenderableEffectDriver> &effectCache);

} // namespace

#endif