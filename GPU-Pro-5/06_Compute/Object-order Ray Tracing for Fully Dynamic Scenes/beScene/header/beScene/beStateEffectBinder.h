/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_STATE_EFFECT_BINDER
#define BE_SCENE_STATE_EFFECT_BINDER

#include "beScene.h"
#include "beEffectBinder.h"
#include "bePassSequence.h"
#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beAPI.h>
#include <beGraphics/Any/beEffectsAPI.h>
#include <beGraphics/Any/beStateManager.h>
#include <lean/smart/resource_ptr.h>
#include <lean/smart/scoped_ptr.h>
#include <lean/containers/dynamic_array.h>
#include <lean/tags/noncopyable.h>

namespace beScene
{

/// State effect binder pass.
class StateEffectBinderPass
{
public:
	/// Shader stage state mask.
	struct ShaderStageStateMask
	{
		bool shaderSet;
		uint4 constantBufferMask;
		uint4 resourceMask;
	};

	/// State mask.
	struct StateMask
	{
		ShaderStageStateMask VSMask;
		ShaderStageStateMask HSMask;
		ShaderStageStateMask DSMask;
		ShaderStageStateMask GSMask;
		ShaderStageStateMask PSMask;
		ShaderStageStateMask CSMask;

		uint4 pipelineMask;
	};

	/// Resource bindings.
	struct ResourceBindings;

private:
	beGraphics::Any::API::EffectPass *m_pPass;
	uint4 m_passID;

	StateMask m_stateMask;
	uint4 m_pipelineRevertMask;

	lean::scoped_ptr<ResourceBindings> m_pResourceBindings;
	uint4 m_controlPointCount;

public:
	/// Constructor.
	BE_SCENE_API StateEffectBinderPass(beGraphics::Any::API::Effect *pEffect, beGraphics::Any::API::EffectPass *pPass, uint4 passID);
	/// Destructor.
	BE_SCENE_API ~StateEffectBinderPass();

	/// Applies the n-th step of this pass.
	BE_SCENE_API bool Apply(beGraphics::Any::StateManager& stateManager, beGraphics::Any::API::DeviceContext *pContext) const;

	/// Gets the input signature of this pass.
	BE_SCENE_API const char* GetInputSignature(uint4 &size) const;

	/// Gets the pass.
	LEAN_INLINE ID3DX11EffectPass* GetPass() const { return m_pPass; }
	/// Gets the pass ID.
	LEAN_INLINE uint4 GetPassID() const { return m_passID; }
};

/// Pipeline effect binder.
class StateEffectBinder : public EffectBinder, public PassSequence<StateEffectBinderPass>
{
public:
	typedef lean::dynamic_array<StateEffectBinderPass> pass_vector;

private:
	const beGraphics::Any::Technique m_technique;

	const pass_vector m_passes;

public:
	/// Constructor.
	BE_SCENE_API StateEffectBinder(const beGraphics::Any::Technique &technique);
	/// Destructor.
	BE_SCENE_API ~StateEffectBinder();

	/// Gets the passes.
	BE_SCENE_API PassRange GetPasses() const;

	/// Gets the technique.
	LEAN_INLINE const beGraphics::Any::Technique& GetTechnique() const { return m_technique; }
	/// Gets the effect.
	LEAN_INLINE const beGraphics::Any::Effect& GetEffect() const { return *m_technique.GetEffect(); }
};

} // namespace

#endif