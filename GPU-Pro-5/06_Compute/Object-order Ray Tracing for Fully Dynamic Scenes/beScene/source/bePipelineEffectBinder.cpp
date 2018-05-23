/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/bePipelineEffectBinder.h"
#include "beScene/beRenderingPipeline.h"
#include <beGraphics/DX/beError.h>
#include <beGraphics/Any/beStateManager.h>

namespace beScene
{

namespace
{

/// Creates passes from the given list of pass IDs.
PipelineEffectBinderPass::pass_vector CreatePasses(beGraphics::Any::API::Effect *pEffect, beGraphics::Any::API::EffectTechnique *pTechnique,
	const uint4 *passIDs, uint4 passCount)
{
	PipelineEffectBinderPass::pass_vector passes(passCount);

	for (uint4 i = 0; i < passCount; ++i)
	{
		uint4 passID = passIDs[i];
		new_emplace(passes) StateEffectBinderPass(pEffect, pTechnique->GetPassByIndex(passID), passID);
	}

	return passes;
}

} // namespace

// Constructor.
PipelineEffectBinderPass::PipelineEffectBinderPass(beGraphics::Any::API::Effect *pEffect, beGraphics::Any::API::EffectTechnique *pTechnique,
												   const uint4 *passIDs, uint4 passCount, uint4 stageID, uint4 queueID)
	: m_passes( CreatePasses(pEffect, pTechnique, passIDs, passCount), lean::consume ),
	m_stageID( stageID ),
	m_queueID( queueID )
{
	LEAN_ASSERT( !m_passes.empty() );
}

// Destructor.
PipelineEffectBinderPass::~PipelineEffectBinderPass()
{
}

// Applies the pass the n-th time.
bool PipelineEffectBinderPass::Apply(uint4 step, beGraphics::Any::StateManager& stateManager, ID3D11DeviceContext *pContext) const
{
	return (step < m_passes.size())
		? m_passes[step].Apply(stateManager, pContext)
		: false;
}

// Gets the n-th step pass, nullptr if unavailable.
const StateEffectBinderPass* PipelineEffectBinderPass::GetPass(uint4 step) const
{
	return (step < m_passes.size())
		? &m_passes[step]
		: nullptr;
}

// Gets the input signature of this pass.
const char* PipelineEffectBinderPass::GetInputSignature(uint4 &size) const
{
	return m_passes[0].GetInputSignature(size);
}

namespace
{

// Gets all passes in the given effect.
PipelineEffectBinder::pass_vector GetPasses(beGraphics::Any::API::Effect *pEffect, beGraphics::Any::API::EffectTechnique *pTechnique,
	RenderingPipeline *pPipeline,
	uint4 binderFlags)
{
	LEAN_ASSERT(pPipeline != nullptr || binderFlags & PipelineEffectBinderFlags::AllowUnclassified);

	PipelineEffectBinder::pass_vector passes;

	D3DX11_TECHNIQUE_DESC techDesc;
	BE_THROW_DX_ERROR_MSG(
		pTechnique->GetDesc(&techDesc),
		"ID3DX11EffectTechnique::GetDesc()");

	struct PassRange
	{
		uint4 stageID;
		uint4 queueID;

		typedef std::vector<uint4> pass_id_vector;
		pass_id_vector passes;

		PassRange(uint4 stageID, uint4 queueID, size_t reservePasses = 0)
			: stageID(stageID),
			queueID(queueID)
		{
			passes.reserve(reservePasses);
		}
	};
	
	typedef std::vector<PassRange> pass_range_vector;
	pass_range_vector ranges;
	ranges.reserve(techDesc.Passes);

	const char *techStageName = "";
	pTechnique->GetAnnotationByName("PipelineStage")->AsString()->GetString(&techStageName);

	const char *techQueueName = "";
	pTechnique->GetAnnotationByName("RenderQueue")->AsString()->GetString(&techQueueName);

	for (UINT passID = 0; passID < techDesc.Passes; ++passID)
	{
		ID3DX11EffectPass *pPass = pTechnique->GetPassByIndex(passID);

		BOOL bNormalPass = true;
		pPass->GetAnnotationByName("Normal")->AsScalar()->GetBool(&bNormalPass);

		// Skip special passes
		if (!bNormalPass)
			continue;

		uint4 stageID = InvalidPipelineStage;
		uint4 queueID = InvalidRenderQueue;

		if (pPipeline)
		{
			// Inherit technique stage, if none specified
			const char *stageName = techStageName;
			pPass->GetAnnotationByName("PipelineStage")->AsString()->GetString(&stageName);
			stageID = pPipeline->GetStageID(stageName);

			// Inherit technique queue, if none specified
			const char *queueName = techQueueName;
			pPass->GetAnnotationByName("RenderQueue")->AsString()->GetString(&queueName);
			queueID = pPipeline->GetQueueID(queueName);

			// Validate classification, if required
			if (~binderFlags & PipelineEffectBinderFlags::AllowUnclassified)
			{
				// Use default stage, if none specified
				if (stageID == InvalidPipelineStage && lean::char_traits<char>::empty(stageName))
					stageID = pPipeline->GetDefaultStageID();

				if (stageID == InvalidPipelineStage)
					LEAN_THROW_ERROR_CTX("Invalid pipeline stage", stageName);

				// Use default queue, if none specified
				if (queueID == InvalidRenderQueue && lean::char_traits<char>::empty(queueName))
					queueID = pPipeline->GetDefaultQueueID(stageID);

				if (queueID == InvalidRenderQueue)
					LEAN_THROW_ERROR_CTX("Invalid render queue", queueName);
			}
		}

		pass_range_vector::iterator itRange = ranges.begin();

		while (itRange != ranges.end() && (itRange->stageID != stageID || itRange->queueID != queueID))
			++itRange;

		if (itRange == ranges.end())
			itRange = ranges.insert( itRange, PassRange(stageID, queueID, techDesc.Passes) );

		itRange->passes.push_back(passID);
	}

	passes.reset(ranges.size());

	for (pass_range_vector::const_iterator itRange = ranges.begin(); itRange != ranges.end(); ++itRange)
		passes.emplace_back(
				pEffect, pTechnique,
				&itRange->passes[0], static_cast<uint4>(itRange->passes.size()), 
				itRange->stageID, itRange->queueID
			);

	return passes;
}

} // namespace

// Constructor.
PipelineEffectBinder::PipelineEffectBinder(const beGraphics::Any::Technique &technique, RenderingPipeline *pPipeline, uint4 flags)
	: m_technique( technique ),
	m_passes( beScene::GetPasses(*m_technique.GetEffect(), m_technique, pPipeline, flags), lean::consume )
{
}

// Destructor.
PipelineEffectBinder::~PipelineEffectBinder()
{
}

// Gets the pass identified by the given ID.
PipelineEffectBinder::PassRange PipelineEffectBinder::GetPasses() const
{
	return beCore::MakeRangeN(&m_passes[0], m_passes.size());
}

} // namespace