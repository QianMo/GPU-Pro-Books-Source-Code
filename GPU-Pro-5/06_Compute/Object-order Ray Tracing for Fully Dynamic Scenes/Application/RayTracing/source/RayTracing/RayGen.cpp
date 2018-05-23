/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "RayTracing/RayGen.h"
#include "RayTracing/RaySet.h"
#include "RayTracing/VoxelRep.h"

#include <beGraphics/Any/beTextureTargetPool.h>
#include <beGraphics/Any/beStateManager.h>

#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beDevice.h>
#include <beGraphics/Any/beDeviceContext.h>

#include <beMath/beVector.h>

#include <beGraphics/DX/beError.h>
#include <lean/logging/errors.h>

namespace app
{

namespace tracing
{

// Binds the given material.
void RayGen::BindMaterial(beg::Material *material)
{
	m_material = material;
	
	m_rayGatherLayer = m_material->GetTechniqueIdx("RayGather");
	m_rayGenLayer = m_material->GetTechniqueIdx("RayGen");
	m_rayGenCohAllocLayer = m_material->GetTechniqueIdx("RayGenCohAlloc");
	m_rayGenCohWriteLayer = m_material->GetTechniqueIdx("RayGenCohWrite");
	m_rayCountLayer = m_material->GetTechniqueIdx("RayCount");
	m_rayOffsetLayer = m_material->GetTechniqueIdx("RayOffset");

	if (m_rayGenLayer == -1 || m_rayGenCohAllocLayer == -1 || m_rayGenCohWriteLayer == -1)
		LEAN_THROW_ERROR_MSG("RayGen technique(s) missing");
	if (m_rayGatherLayer == -1)
		LEAN_THROW_ERROR_MSG("RayGather technique missing");
	if (m_rayCountLayer == -1 || m_rayOffsetLayer == -1)
		LEAN_THROW_ERROR_MSG("Ray z-order layout technique(s) missing");

	m_effect = ToImpl(m_material->GetEffects()[0])->Get();

	m_voxelRepConstVar = beg::Any::ValidateEffectVariable(m_effect->GetConstantBufferByName("VoxelRepConstants"), LSS);
	m_raySetConstVar = beg::Any::ValidateEffectVariable(m_effect->GetConstantBufferByName("RaySetConstants"), LSS);

	m_rayQueueUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayQueueUAV")->AsUnorderedAccessView(), LSS);
	m_rayDescVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayDescriptions")->AsShaderResource(), LSS);
	m_tracedGeometryVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("TracedGeometry")->AsShaderResource(), LSS);
	m_tracedLightVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("TracedLight")->AsShaderResource(), LSS);

	m_gridDebugVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("DebugGrid")->AsShaderResource(), LSS);
	m_rayDebugVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("DebugRays")->AsShaderResource(), LSS);

	m_rayCountVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayCountTexture")->AsShaderResource(), LSS);
	m_rayOffsetVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayOffsetTexture")->AsShaderResource(), LSS);
	
	m_processor->SetMaterial(m_material);
}

// Constructor.
RayGen::RayGen(const lean::utf8_ntri &file, besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager)
	: m_renderer( renderer ),
	
	m_processor( new_resource besc::QuadProcessor(renderer->Device(), renderer->ProcessingDrivers()) )
{
	BindMaterial( resourceManager->MaterialCache()->NewByFile(file, "RayGen") );
}

// Destructor.
RayGen::~RayGen()
{
}

// Processes / commits changes.
void RayGen::Commit()
{
	// Hot swap
	if (beg::Material *successor = m_material->GetSuccessor())
		BindMaterial(successor);
}

// Performs ray compaction.
void RayGen::RayCompact(besc::Perspective &perspective, besc::RenderContext &context) const
{
	beg::api::DeviceContext *deviceContext = ToImpl(context.Context());
	besc::Pipe &pipe = *perspective.GetPipe();

	lean::com_ptr<const beg::ColorTextureTarget> rayCountTarget = pipe.GetColorTarget("RayCountTarget");

	if (!rayCountTarget)
	{
		LEAN_LOG_ERROR_CTX("Ray count target missing", "RayCountTarget");
		return;
	}

	beg::TextureTargetDesc desc = FromAPI(rayCountTarget->GetDesc());
	beg::TextureTargetDesc mipDesc = desc;
	mipDesc.MipLevels = 1;
	
	bem::vector<uint4, 2> levelRes[32];
	levelRes[0] = bem::vec(desc.Width, desc.Height);
	lean::com_ptr<const beg::ColorTextureTarget> levelTargets[32];
	levelTargets[0] = rayCountTarget;
	uint4 level = 0;

	static const uint4 ResolutionStep = 2;
	static const uint4 LevelStep = 1;

	ToImpl(context.StateManager()).Revert();
	ToImpl(context.StateManager()).Reset();

	while (max_scan(levelRes[level]) > ResolutionStep)
	{
		level += LevelStep;
		mipDesc.Width = levelRes[level].x = (levelRes[level - LevelStep].x + ResolutionStep - 1) / ResolutionStep;
		mipDesc.Height = levelRes[level].y = (levelRes[level - LevelStep].y + ResolutionStep - 1) / ResolutionStep;

		levelTargets[level] = pipe.GetNewColorTarget("RayCountTarget", mipDesc, 0, 0);
		D3D11_VIEWPORT vp = { 0.0f, 0.0f, (float) mipDesc.Width, (float) mipDesc.Height, 0.0f, 1.0f };

		beg::api::RenderTargetView *mipTarget = levelTargets[level]->GetTarget();
		beg::api::ShaderResourceView *mipTexture = levelTargets[level - LevelStep]->GetTexture();

		m_rayCountVar->SetResource(mipTexture);

		deviceContext->OMSetRenderTargets(1, &mipTarget, nullptr);
		deviceContext->RSSetViewports(1, &vp);

		m_processor->Render(m_rayCountLayer, &perspective, context);
	}

	lean::com_ptr<const beg::ColorTextureTarget> rayIndexTarget, rayIndexBase;

	while (level != -1)
	{
		mipDesc.Width = levelRes[level].x;
		mipDesc.Height = levelRes[level].y;

		rayIndexTarget = pipe.GetNewColorTarget("RayOffsetTarget", mipDesc, besc::PipeTargetFlags::Persistent, 0);
		D3D11_VIEWPORT vp = { 0.0f, 0.0f, (float) mipDesc.Width, (float) mipDesc.Height, 0.0f, 1.0f };

		beg::api::RenderTargetView *mipTarget = rayIndexTarget->GetTarget();
		beg::api::ShaderResourceView *mipTexture = (rayIndexBase) ? rayIndexBase->GetTexture() : nullptr;
		beg::api::ShaderResourceView *countMipTexture = levelTargets[level]->GetTexture();

		m_rayOffsetVar->SetResource(mipTexture);
		m_rayCountVar->SetResource(countMipTexture);

		deviceContext->OMSetRenderTargets(1, &mipTarget, nullptr);
		deviceContext->RSSetViewports(1, &vp);

		m_processor->Render(m_rayOffsetLayer, &perspective, context);

		rayIndexBase = rayIndexTarget;
		level -= LevelStep;
	}

	beg::Any::UnbindAllRenderTargets(deviceContext);
}

// Generates rays for the given bundle.
void RayGen::GenerateRaysCoherent(besc::Perspective &perspective, RaySet &raySet, 
	bool bKeepRays, besc::RenderContext &context) const
{
	m_raySetConstVar->SetConstantBuffer(raySet.Constants());
	m_rayQueueUAVVar->SetUnorderedAccessView(raySet.RayDescUAV(), (bKeepRays) ? -1 : 0);

	m_processor->Render(m_rayGenCohAllocLayer, &perspective, context);
	beg::Any::UnbindAllRenderTargets(ToImpl(context.Context()));

	RayCompact(perspective, context);

	m_processor->Render(m_rayGenCohWriteLayer, &perspective, context);
	beg::Any::UnbindAllRenderTargets(ToImpl(context.Context()));
}

// Generates rays for the given bundle.
void RayGen::GenerateRays(besc::Perspective &perspective, RaySet &raySet, 
	bool bKeepRays, besc::RenderContext &context) const
{
	m_raySetConstVar->SetConstantBuffer(raySet.Constants());
	m_rayQueueUAVVar->SetUnorderedAccessView(raySet.RayDescUAV(), (bKeepRays) ? -1 : 0);

	m_processor->Render(m_rayGenLayer, &perspective, context);

	beg::Any::UnbindAllRenderTargets(ToImpl(context.Context()));
}

// Gathers rays.
void RayGen::GatherRays(besc::Perspective &perspective, RaySet &raySet, VoxelRep &voxelRep,
		beg::api::ShaderResourceView *debugGrid, besc::RenderContext &context) const
{
	m_voxelRepConstVar->SetConstantBuffer(voxelRep.Constants());
	m_raySetConstVar->SetConstantBuffer(raySet.Constants());
	m_tracedGeometryVar->SetResource(raySet.RayGeometry());
	m_tracedLightVar->SetResource(raySet.RayLight());
	m_rayDescVar->SetResource(raySet.RayDescs());

	m_rayDebugVar->SetResource(raySet.RayDebug());
	m_gridDebugVar->SetResource(debugGrid);

	m_processor->Render(m_rayGatherLayer, &perspective, context);
}

} // namespace

} // namespace
