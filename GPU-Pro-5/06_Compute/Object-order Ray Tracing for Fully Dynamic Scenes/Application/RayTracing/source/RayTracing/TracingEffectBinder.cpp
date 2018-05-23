/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "RayTracing/TracingEffectBinder.h"
#include "RayTracing/TracingEffectBinderPool.h"
#include "RayTracing/Pipeline.h"
#include "RayTracing/RaySet.h"

#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beBuffer.h>
#include <beGraphics/Any/beTexture.h>
#include <beGraphics/DX/beError.h>
#include <beMath/beVector.h>

#include <lean/limits.h>
#include <lean/io/numeric.h>

namespace app
{

namespace tracing
{

/// Pass.
struct TracingEffectBinder::Pass
{
	beg::api::EffectPass *pass;
	beg::api::EffectPass *pIntersectCellRaysPass;
	beg::api::EffectPass *pCSGroupPass;

	bool bDynamicDispatch;
	bool bSkipIntersection;
};

namespace
{

/// Gets a scalar variable of the given name or nullptr, if unavailable.
ID3DX11EffectScalarVariable* MaybeGetScalarVariable(ID3DX11Effect *pEffect, const lean::utf8_ntri &name)
{
	ID3DX11EffectScalarVariable *pVariable = pEffect->GetVariableBySemantic(name.c_str())->AsScalar();
	return (pVariable->IsValid()) ? pVariable : nullptr;
}

/// Gets a shader resource variable of the given name or nullptr, if unavailable.
ID3DX11EffectShaderResourceVariable* MaybeGetResourceVariable(ID3DX11Effect *pEffect, const lean::utf8_ntri &name)
{
	ID3DX11EffectShaderResourceVariable *pVariable = pEffect->GetVariableBySemantic(name.c_str())->AsShaderResource();
	return (pVariable->IsValid()) ? pVariable : nullptr;
}

/// Gets a pass variable of the given name or nullptr, if unavailable.
ID3DX11EffectPass* MaybeGetPass(ID3DX11EffectTechnique *technique, const lean::utf8_ntri &name)
{
	ID3DX11EffectPass *pass = technique->GetPassByName(name.c_str());
	return (pass->IsValid()) ? pass : nullptr;
}

/// Gets the given pass from the given technique.
TracingEffectBinder::Pass GetPass(ID3DX11EffectTechnique *technique, UINT passID)
{
	TracingEffectBinder::Pass pass;

	beGraphics::Any::API::EffectPass *passDX = technique->GetPassByIndex(passID);

	if (!passDX->IsValid())
		LEAN_THROW_ERROR_MSG("ID3DX11Technique::GetPassByIndex()");

	pass.pass = passDX;

	const char *intersectCellRaysPass = "";
	passDX->GetAnnotationByName("IntersectCellRaysPass")->AsString()->GetString(&intersectCellRaysPass);
	pass.pIntersectCellRaysPass = MaybeGetPass(technique, intersectCellRaysPass);

	const char *csGroupPass = "";
	passDX->GetAnnotationByName("TracingCSGroupPass")->AsString()->GetString(&csGroupPass);
	pass.pCSGroupPass = MaybeGetPass(technique, csGroupPass);

	BOOL bDynamicDispatch = 0;
	if (pass.pIntersectCellRaysPass)
		pass.pIntersectCellRaysPass->GetAnnotationByName("DynamicDispatch")->AsScalar()->GetBool(&bDynamicDispatch);
	pass.bDynamicDispatch = (bDynamicDispatch != 0);

	BOOL bSkipIntersection = 0;
	if (pass.pIntersectCellRaysPass)
		pass.pIntersectCellRaysPass->GetAnnotationByName("NullSkip")->AsScalar()->GetBool(&bSkipIntersection);
	pass.bSkipIntersection = (bSkipIntersection != 0);
	
	if (pass.pIntersectCellRaysPass && !(pass.bDynamicDispatch || pass.pCSGroupPass))
		LEAN_THROW_ERROR_MSG("Missing tracing CS group pass");

	return pass;
}

/// Gets all passes in the given technique.
TracingEffectBinder::pass_vector GetPasses(ID3DX11EffectTechnique *technique, uint4 singlePassID = static_cast<uint4>(-1))
{
	TracingEffectBinder::pass_vector passes;

	D3DX11_TECHNIQUE_DESC techniqueDesc;
	BE_THROW_DX_ERROR_MSG(
		technique->GetDesc(&techniqueDesc),
		"ID3DX11Technique::GetDesc()");
	
	if (singlePassID < techniqueDesc.Passes)
		// Load single pass
		passes.push_back( GetPass(technique, singlePassID) );
	else
	{
		passes.reserve(techniqueDesc.Passes);

		// Load all passes
		for (UINT passID = 0; passID < techniqueDesc.Passes; ++passID)
			passes.push_back( GetPass(technique, passID) );
	}

	return passes;
}

} // namespace

// Constructor.
TracingEffectBinder::TracingEffectBinder(const beGraphics::Any::Technique &technique, TracingEffectBinderPool *pool, uint4 passID)
	: m_technique( technique ),
	m_passes( GetPasses(m_technique, passID) ),
	m_pool( LEAN_ASSERT_NOT_NULL(pool) ),

	m_tracingConstantsVar( technique.GetEffect()->Get()->GetConstantBufferByName("TracingConstants") ),

	m_tracedGeometryUAV( technique.GetEffect()->Get()->GetVariableBySemantic("TracedGeometryUAV")->AsUnorderedAccessView() ),
	m_tracedLightUAV( technique.GetEffect()->Get()->GetVariableBySemantic("TracedLightUAV")->AsUnorderedAccessView() ),
	m_debugUAV( technique.GetEffect()->Get()->GetVariableBySemantic("DebugUAV")->AsUnorderedAccessView() ),

	m_triangleSRVVar( technique.GetEffect()->Get()->GetVariableBySemantic("TracingTriangles")->AsShaderResource() ),
	m_geometrySRVVar( technique.GetEffect()->Get()->GetVariableBySemantic("TracingGeometry")->AsShaderResource() ),

	m_voxelInVar( technique.GetEffect()->Get()->GetVariableBySemantic("TracingVoxelIn")->AsShaderResource() ),
	m_voxelOutVar( technique.GetEffect()->Get()->GetVariableBySemantic("TracingVoxelOut")->AsUnorderedAccessView() ),
	
	m_groupDispatchUAV( technique.GetEffect()->Get()->GetVariableBySemantic("GroupDispatchUAV")->AsUnorderedAccessView() )
{
}

// Destructor.
TracingEffectBinder::~TracingEffectBinder()
{
}

// Applies stuff.
bool TracingEffectBinder::Apply(beGraphics::Any::API::DeviceContext *pContext) const
{
	// Common bindings
	RaySet &raySet = *LEAN_ASSERT_NOT_NULL(m_pool->GetRaySet());
	m_tracedGeometryUAV->SetUnorderedAccessView( raySet.RayGeometryUAV(), -1 );
	m_tracedLightUAV->SetUnorderedAccessView( raySet.RayLightUAV(), -1 );
	m_debugUAV->SetUnorderedAccessView( m_pool->GetDebugUAV(), -1 );
	return true;
}

// Computes the number of groups to dispatch.
void TracingEffectBinder::ComputeDispatchGroupCount(beg::api::EffectPass *pass, beg::api::UnorderedAccessView *dispatchUAV, beg::api::DeviceContext *context) const
{
	m_groupDispatchUAV->SetUnorderedAccessView(dispatchUAV);

	pass->Apply(0, context);
	context->Dispatch(1, 1, 1);

	pass->Unbind(context);
}

// Draws the given number of primitives.
bool TracingEffectBinder::Render(uint4 primitiveCount, lean::vcallable<besc::AbstractRenderableEffectDriver::DrawJobSignature> &drawJob,
	uint4 passID, beGraphics::Any::StateManager &stateManager, beg::api::DeviceContext *context) const
{
	// Ignore end of pass range
	if (passID >= m_passes.size())
		return false;

	const Pass &pass = m_passes[passID];

	if (pass.pIntersectCellRaysPass)
	{
		TracingEffectBinderPool::IntermediateBuffer triangles = m_pool->GetTriangleBuffer(primitiveCount);
		// TODO: Get stride from somewhere
		TracingEffectBinderPool::IntermediateBuffer geometry = m_pool->GetGeometryBuffer(60 * primitiveCount);
		TracingEffectBinderPool::IntermediateBuffer voxels = m_pool->GetVoxelBuffer(primitiveCount);
		
		if (triangles.buffer && geometry.buffer && voxels.buffer)
		{
			D3D11_PRIMITIVE_TOPOLOGY topology;
			lean::com_ptr<ID3D11InputLayout> inputLayout;
			context->IAGetPrimitiveTopology(&topology);
			context->IAGetInputLayout(inputLayout.rebind());

			// Common bindings
			RaySet &raySet = *LEAN_ASSERT_NOT_NULL(m_pool->GetRaySet());
			m_tracedGeometryUAV->SetUnorderedAccessView( raySet.RayGeometryUAV(), -1 );
			m_tracedLightUAV->SetUnorderedAccessView( raySet.RayLightUAV(), -1 );
			m_debugUAV->SetUnorderedAccessView( m_pool->GetDebugUAV(), -1 );
			
			// Stream out geometry & voxel-triangle pairs
			{
				m_voxelOutVar->SetUnorderedAccessView(voxels.uav, 0);

				beg::api::Buffer *const geometryBuffers[] = { triangles.buffer, geometry.buffer };
				const uint4 geometryBufferOffset[2] =  { 0 };
				context->SOSetTargets(lean::arraylen(geometryBuffers), geometryBuffers, geometryBufferOffset);
				
				// Re-apply to activate UAV ... TODO: Better effect framework / shader interface
				pass.pass->Apply(0, context);
				drawJob(passID, stateManager, beg::Any::DeviceContext(context));

				pass.pass->Unbind(context);
				context->SOSetTargets(0, nullptr, nullptr);

//				float triData[1024];
//				beg::Any::DebugFetchBufferData(context, geometry.buffer, triData, sizeof(triData));
			}

			// Do intersection testing
			if (!pass.bSkipIntersection)
			{
				// Bindings
				m_geometrySRVVar->SetResource(geometry.srv);
				m_triangleSRVVar->SetResource(triangles.srv);
				m_voxelInVar->SetResource(voxels.srv);

				// Geometry shader dynamic dispatch path
				if (pass.bDynamicDispatch)
				{
					TracingEffectBinderPool::IntermediateBuffer drawIndirectBuffer = m_pool->GetSingleInstanceBuffer();

					// ORDER: WORKAROUND: AMD driver missing UAV counter dependency chain
					pass.pIntersectCellRaysPass->Apply(0, context);
					
					context->CopyStructureCount(drawIndirectBuffer.buffer, 0, voxels.uav);

					context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
					context->IASetInputLayout(m_noInputLayout);
					
					// 'Draw' one vertex per buffer entry
					context->DrawInstancedIndirect(drawIndirectBuffer.buffer, 0);
					
					context->IASetPrimitiveTopology(topology);
					context->IASetInputLayout(inputLayout);
				}
				// Compute shader path
				else
				{
					TracingEffectBinderPool::IntermediateBuffer constantBuffer = m_pool->GetConstantBuffer();
					TracingEffectBinderPool::IntermediateBuffer dispatchBuffer = m_pool->GetDispatchBuffer();

					// ORDER: WORKAROUND: AMD driver missing UAV counter dependency chain
					pass.pIntersectCellRaysPass->Apply(0, context);

					context->CopyStructureCount(constantBuffer.buffer, offsetof(TracingBinderConstants, InputCount), voxels.uav);
					m_tracingConstantsVar->SetConstantBuffer(constantBuffer.buffer);

					ComputeDispatchGroupCount(pass.pCSGroupPass, dispatchBuffer.uav, context);

					// Process grid rays
					pass.pIntersectCellRaysPass->Apply(0, context);
					context->DispatchIndirect(dispatchBuffer.buffer, 0);
				}

				pass.pIntersectCellRaysPass->Unbind(context);
			}
		}

		return true;
	}
	// Simply dispatch compute shader w/ custom number of groups
	else if (pass.pCSGroupPass)
	{
		RaySet &raySet = *LEAN_ASSERT_NOT_NULL(m_pool->GetRaySet());
		TracingEffectBinderPool::IntermediateBuffer dispatchBuffer = m_pool->GetDispatchBuffer();

		// Common bindings
		m_tracedGeometryUAV->SetUnorderedAccessView( raySet.RayGeometryUAV(), -1 );
		m_tracedLightUAV->SetUnorderedAccessView( raySet.RayLightUAV(), -1 );
		m_debugUAV->SetUnorderedAccessView( m_pool->GetDebugUAV(), -1 );

		// Compute custonumber of groups
		ComputeDispatchGroupCount(pass.pCSGroupPass, dispatchBuffer.uav, context);

		pass.pass->Apply(0, context);
		context->DispatchIndirect(dispatchBuffer.buffer, 0);

		pass.pass->Unbind(context);

		return true;
	}
	// Ignore non-tracing pass
	else
		return false;
}

} // namespace

} // namespace
