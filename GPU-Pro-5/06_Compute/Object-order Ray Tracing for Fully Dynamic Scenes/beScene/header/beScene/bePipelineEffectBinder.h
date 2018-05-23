/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PIPELINE_EFFECT_BINDER
#define BE_SCENE_PIPELINE_EFFECT_BINDER

#include "beScene.h"
#include "beEffectBinder.h"
#include "bePassSequence.h"
#include "beQueuedPass.h"
#include "beStateEffectBinder.h"
#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beAPI.h>
#include <beGraphics/Any/beEffectsAPI.h>
#include <beGraphics/Any/beStateManager.h>
#include <lean/containers/dynamic_array.h>

namespace beScene
{

/// Pipeline effect binder flags enumeration.
struct PipelineEffectBinderFlags
{
	/// Enumeration.
	enum T
	{
		AllowUnclassified = 0x1		///< Permits passes & technique to omit stage & queue classification.
	};
	LEAN_MAKE_ENUM_STRUCT(PipelineEffectBinderFlags)
};

// Prototypes
class RenderingPipeline;

/// Pipeline effect binder pass.
class PipelineEffectBinderPass : public QueuedPass
{
public:
	typedef lean::dynamic_array<StateEffectBinderPass> pass_vector;

private:
	const pass_vector m_passes;

	uint4 m_stageID;
	uint4 m_queueID;

public:
	/// Constructor.
	BE_SCENE_API PipelineEffectBinderPass(beGraphics::Any::API::Effect *pEffect, beGraphics::Any::API::EffectTechnique *pTechnique,
		const uint4 *passIDs, uint4 passCount, uint4 stageID, uint4 queueID);
	/// Destructor.
	BE_SCENE_API ~PipelineEffectBinderPass();

	/// Applies the n-th step of this pass.
	BE_SCENE_API bool Apply(uint4 step, beGraphics::Any::StateManager& stateManager, beGraphics::Any::API::DeviceContext *pContext) const;
	
	/// Gets the n-th step pass, nullptr if unavailable.
	BE_SCENE_API const StateEffectBinderPass* GetPass(uint4 step) const;

	/// Gets the input signature of this pass.
	BE_SCENE_API const char* GetInputSignature(uint4 &size) const;

	/// Gets the stage ID.
	LEAN_INLINE uint4 GetStageID() const { return m_stageID; }
	/// Gets the queue ID.
	LEAN_INLINE uint4 GetQueueID() const { return m_queueID; }
};

/// Pipeline effect binder.
class PipelineEffectBinder : public EffectBinder, public PassSequence<PipelineEffectBinderPass>
{
public:
	typedef lean::dynamic_array<PipelineEffectBinderPass> pass_vector;

private:
	const beGraphics::Any::Technique m_technique;
	
	const pass_vector m_passes;

public:
	/// Constructor.
	BE_SCENE_API PipelineEffectBinder(const beGraphics::Any::Technique &technique, RenderingPipeline *pPipeline, uint4 flags = 0);
	/// Destructor.
	BE_SCENE_API ~PipelineEffectBinder();

	/// Gets the passes.
	BE_SCENE_API PassRange GetPasses() const;

	/// Gets the technique.
	LEAN_INLINE const beGraphics::Technique& GetTechnique() const { return m_technique; }
	/// Gets the effect.
	LEAN_INLINE const beGraphics::Effect& GetEffect() const { return *m_technique.GetEffect(); }
};

} // namespace

#endif