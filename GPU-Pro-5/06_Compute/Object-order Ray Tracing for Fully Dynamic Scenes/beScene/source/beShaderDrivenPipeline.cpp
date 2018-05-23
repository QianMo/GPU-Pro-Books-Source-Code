/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beShaderDrivenPipeline.h"
#include "beScene/beRenderingPipeline.h"
#include "beScene/beAbstractRenderableEffectDriver.h"
#include "beScene/beEffectBinderCache.h"
#include "beScene/beEffectQueueSetup.h"
#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beEffectsAPI.h>
#include <beGraphics/DX/beError.h>

using namespace beGraphics;

namespace beScene
{

namespace
{

/// Tries to read the description of the given stage from the given effect.
PipelineStageDesc GetStageDesc(ID3DX11Effect *pEffect, ID3DX11EffectVariable *pStage)
{
	int4 layer;
	BE_THROW_DX_ERROR_MSG( pStage->GetAnnotationByName("Layer")->AsScalar()->GetInt(&layer), "ID3DX11EffectScalarVariable::GetInt()" );

	BOOL bNormal = TRUE;
	pStage->GetAnnotationByName("Normal")->AsScalar()->GetBool(&bNormal);

	BOOL bConditional = FALSE;
	pStage->GetAnnotationByName("Conditional")->AsScalar()->GetBool(&bConditional);

	return PipelineStageDesc(layer, bNormal != FALSE, nullptr, bConditional != FALSE);
}

/// Tries to read the description of the given stage from the given effect.
PipelineStageDesc GetStageDesc(ID3DX11Effect *pEffect, const utf8_ntri &stageName)
{
	ID3DX11EffectVariable *pStage = pEffect->GetVariableByName(stageName.c_str());

	if (!pStage->IsValid())
		LEAN_THROW_ERROR_CTX("ID3DX11Effect::GetVariableByName()", stageName.c_str());

	return GetStageDesc(pEffect, pStage);
}

/// Tries to read the description of the given queue from the given effect.
RenderQueueDesc GetQueueDesc(ID3DX11Effect *pEffect, ID3DX11EffectVariable *pQueue)
{
	int4 layer;
	BE_THROW_DX_ERROR_MSG( pQueue->GetAnnotationByName("Layer")->AsScalar()->GetInt(&layer), "ID3DX11EffectScalarVariable::GetInt()" );

	BOOL bDepthSort = FALSE;
	pQueue->GetAnnotationByName("DepthSort")->AsScalar()->GetBool(&bDepthSort);

	BOOL bBackwards = FALSE;
	pQueue->GetAnnotationByName("Backwards")->AsScalar()->GetBool(&bBackwards);

	BOOL bConditional = TRUE;
	pQueue->GetAnnotationByName("Conditional")->AsScalar()->GetBool(&bConditional);

	return RenderQueueDesc(layer, bDepthSort != FALSE, bBackwards != FALSE, nullptr, bConditional != FALSE);
}

/// Tries to read the description of the given queue from the given effect.
RenderQueueDesc GetQueueDesc(ID3DX11Effect *pEffect, const utf8_ntri &queueName)
{
	ID3DX11EffectVariable *pQueue = pEffect->GetVariableByName(queueName.c_str());

	if (!pQueue->IsValid())
		LEAN_THROW_ERROR_CTX("ID3DX11Effect::GetVariableByName()", queueName.c_str());

	return GetQueueDesc(pEffect, pQueue);
}

/// Tries to read the description of the given queue from the given effect.
lean::resource_ptr<const QueueSetup, true> GetQueueSetup(const beg::Any::Effect *effect, ID3DX11EffectVariable *pQueue,
	EffectDriverCache<AbstractRenderableEffectDriver> &effectCache)
{
	const char *setupTechniqueName = nullptr;
	pQueue->GetAnnotationByName("Setup")->AsString()->GetString(&setupTechniqueName);

	lean::resource_ptr<QueueSetup> pSetup;

	if (setupTechniqueName && !lean::char_traits<char>::empty(setupTechniqueName))
	{
		Any::API::EffectTechnique *pSetupTechnique = effect->Get()->GetTechniqueByName(setupTechniqueName);

		if (pSetupTechnique->IsValid())
		{
			AbstractRenderableEffectDriver *pSetupDriver = effectCache.GetEffectBinder(
				Any::Technique(effect, pSetupTechnique),
				RenderableEffectDriverFlags::Setup );
			pSetup = new_resource EffectQueueSetup(pSetupDriver);
		}
		else
			LEAN_LOG_ERROR_CTX("Invalid setup technique", setupTechniqueName);
	}

	return pSetup.transfer();
}

/// Tries to read the description of the given queue from the given effect.
lean::resource_ptr<const QueueSetup, true> GetQueueSetup(const beg::Any::Effect *effect, const utf8_ntri &queueName,
	EffectDriverCache<AbstractRenderableEffectDriver> &effectCache)
{
	ID3DX11EffectVariable *pQueue = effect->Get()->GetVariableByName(queueName.c_str());

	if (!pQueue->IsValid())
		LEAN_THROW_ERROR_CTX("ID3DX11Effect::GetVariableByName()", queueName.c_str());

	return GetQueueSetup(effect, pQueue, effectCache);
}

} // namespace

// Loads all pipeline stages from the given effect.
void LoadPipelineStages(RenderingPipeline &pipeline, const beg::Effect *effect,
	EffectDriverCache<AbstractRenderableEffectDriver> &effectCache)
{
	const beg::Any::Effect *effectImpl = ToImpl(effect);
	beg::api::Effect *effectDX = effectImpl->Get();
	D3DX11_EFFECT_DESC effectDesc = GetDesc(effectDX);

	// Load stages
	for (UINT variableID = 0; variableID < effectDesc.GlobalVariables; ++variableID)
	{
		Any::API::EffectVariable *pVariable = effectDX->GetVariableByIndex(variableID);
		const char *typeName = GetDesc(pVariable->GetType()).TypeName;

		if (lean::char_traits<char>::equal(typeName, "PipelineStage"))
			pipeline.AddStage(
					GetDesc(pVariable).Name,
					GetStageDesc(effectDX, pVariable)
				);
	}

	// Load stage setups
	// -> Mixed setup techniques might require all stages to be known in advance
	for (UINT variableID = 0; variableID < effectDesc.GlobalVariables; ++variableID)
	{
		Any::API::EffectVariable *pVariable = effectDX->GetVariableByIndex(variableID);
		const char *typeName = GetDesc(pVariable->GetType()).TypeName;

		if (lean::char_traits<char>::equal(typeName, "PipelineStage"))
			pipeline.SetStageSetup(
					pipeline.GetStageID(GetDesc(pVariable).Name),
					GetQueueSetup(effectImpl, pVariable, effectCache).get()
				);
	}
}

// Loads all render queues from the given effect.
void LoadRenderQueues(RenderingPipeline &pipeline, const Effect *effect,
	EffectDriverCache<AbstractRenderableEffectDriver> &effectCache)
{
	const Any::Effect *effectImpl = ToImpl(effect);
	API::Effect *effectDX = effectImpl->Get();
	D3DX11_EFFECT_DESC effectDesc = GetDesc(effectDX);

	// Load queues
	for (UINT variableID = 0; variableID < effectDesc.GlobalVariables; ++variableID)
	{
		Any::API::EffectVariable *pVariable = effectDX->GetVariableByIndex(variableID);
		const char *typeName = GetDesc(pVariable->GetType()).TypeName;

		if (lean::char_traits<char>::equal(typeName, "RenderQueue"))
			pipeline.AddQueue(
					GetDesc(pVariable).Name,
					GetQueueDesc(effectDX, pVariable)
				);
	}
	
	// Load queue setups
	// -> Mixed setup techniques might require all queues to be known in advance
	for (UINT variableID = 0; variableID < effectDesc.GlobalVariables; ++variableID)
	{
		Any::API::EffectVariable *pVariable = effectDX->GetVariableByIndex(variableID);
		const char *typeName = GetDesc(pVariable->GetType()).TypeName;

		if (lean::char_traits<char>::equal(typeName, "RenderQueue"))
			pipeline.SetQueueSetup(
					pipeline.GetQueueID(GetDesc(pVariable).Name),
					GetQueueSetup(effectImpl, pVariable, effectCache).get()
				);
	}
}

// Loads all pipeline information from the given effect.
void LoadRenderingPipeline(RenderingPipeline &pipeline, const beGraphics::Effect *effect,
	EffectDriverCache<AbstractRenderableEffectDriver> &effectCache)
{
	LoadPipelineStages(pipeline, effect, effectCache);
	LoadRenderQueues(pipeline, effect, effectCache);
}

} // namespace