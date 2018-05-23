/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beStateEffectBinder.h"
#include "beScene/beRenderingPipeline.h"
#include <beGraphics/DX/beError.h>
#include <beGraphics/Any/beStateManager.h>

#include <vector>
#include <lean/smart/com_ptr.h>

using namespace beGraphics;

namespace beScene
{

struct ResourceBindingType
{
	enum T
	{
		ConstantBuffer,
		Resource
	};
	LEAN_MAKE_ENUM_STRUCT(ResourceBindingType)
};

/// Resource binding.
struct ResourceBinding
{
	ResourceBindingType::T type;
	void *pResource;
	uint4 bindPoint;

	/// Constant buffer resource binding constructor.
	ResourceBinding(Any::API::EffectConstantBuffer *pConstants, uint4 bindPoint)
		: type(ResourceBindingType::ConstantBuffer),
		pResource(pConstants),
		bindPoint(bindPoint) { }
	/// Shader resource binding constructor.
	ResourceBinding(Any::API::EffectShaderResource *pConstants, uint4 bindPoint)
		: type(ResourceBindingType::Resource),
		pResource(pConstants),
		bindPoint(bindPoint) { }
};

// Resource bindings.
struct StateEffectBinderPass::ResourceBindings
{
	typedef std::vector<ResourceBinding> binding_vector;

	binding_vector VSResourceBindings;
	binding_vector HSResourceBindings;
	binding_vector DSResourceBindings;
	binding_vector GSResourceBindings;
	binding_vector PSResourceBindings;
	binding_vector CSResourceBindings;
};

namespace
{

/// Computes a bit mask from the given bit array.
template <class Type, class Element>
inline Type BitArrayToMask(const Element *bits, size_t bitCount)
{
	Type mask = 0;

	const size_t maskBitCount = min(lean::size_info<Type>::bits, bitCount);

	for (size_t i = 0; i < maskBitCount; ++i)
		mask |= static_cast<Type>( (bits[i / lean::size_info<Element>::bits] >> (i % lean::size_info<Element>::bits)) & 0x1U ) << i;

	return mask;
}

/// Computes a shader stage state mask from the given DirectX 11 state masks.
StateEffectBinderPass::ShaderStageStateMask GetShaderStageStateMask(BYTE shaderSet, const BYTE *constantBufferMask, const BYTE *resourceMask)
{
	StateEffectBinderPass::ShaderStageStateMask mask;
	
	mask.shaderSet = (shaderSet != 0);
	mask.constantBufferMask = BitArrayToMask<uint4>(constantBufferMask, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT);
	mask.resourceMask = BitArrayToMask<uint4>(resourceMask, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT);

	return mask;
}

/// Computes a state mask for the given pass.
StateEffectBinderPass::StateMask GetStateMask(Any::API::EffectPass *pPass)
{
	StateEffectBinderPass::StateMask mask;

	// NOTE: ComputeStateBlockMask does not initialize to 0
	D3DX11_STATE_BLOCK_MASK stateBlock = { 0 };
	BE_THROW_DX_ERROR_MSG(
		pPass->ComputeStateBlockMask(&stateBlock),
		"ID3DX11EffectTechnique::ComputeStateBlockMask()" );

	mask.VSMask = GetShaderStageStateMask(stateBlock.VS, stateBlock.VSConstantBuffers, stateBlock.VSShaderResources);
	mask.HSMask = GetShaderStageStateMask(stateBlock.HS, stateBlock.HSConstantBuffers, stateBlock.HSShaderResources);
	mask.DSMask = GetShaderStageStateMask(stateBlock.DS, stateBlock.DSConstantBuffers, stateBlock.DSShaderResources);
	mask.GSMask = GetShaderStageStateMask(stateBlock.GS, stateBlock.GSConstantBuffers, stateBlock.GSShaderResources);
	mask.PSMask = GetShaderStageStateMask(stateBlock.PS, stateBlock.PSConstantBuffers, stateBlock.PSShaderResources);
	mask.CSMask = GetShaderStageStateMask(stateBlock.CS, stateBlock.CSConstantBuffers, stateBlock.CSShaderResources);

	mask.pipelineMask = 0;

	if (stateBlock.RSRasterizerState != 0)
		mask.pipelineMask |= DX11::StateMasks::RasterizerState;
	if (stateBlock.OMDepthStencilState != 0)
		mask.pipelineMask |= DX11::StateMasks::DepthStencilState;
	if (stateBlock.OMBlendState != 0)
		mask.pipelineMask |= DX11::StateMasks::BlendState;

	return mask;
}

/// Computes a state revert mask for the given pass.
uint4 GetStateRevertMask(Any::API::EffectPass *pPass)
{
	uint4 revertMask = 0;

	BOOL bRevertRasterizer = FALSE;
	pPass->GetAnnotationByName("RevertRasterizer")->AsScalar()->GetBool(&bRevertRasterizer);
	if (bRevertRasterizer != FALSE)
		revertMask |= DX11::StateMasks::RasterizerState;

	BOOL bRevertDepthStencil = FALSE;
	pPass->GetAnnotationByName("RevertDepthStencil")->AsScalar()->GetBool(&bRevertDepthStencil);
	if (bRevertDepthStencil != FALSE)
		revertMask |= DX11::StateMasks::DepthStencilState;

	BOOL bRevertBlending = FALSE;
	pPass->GetAnnotationByName("RevertBlending")->AsScalar()->GetBool(&bRevertBlending);
	if (bRevertBlending != FALSE)
		revertMask |= DX11::StateMasks::BlendState;

	BOOL bRevertTargets = FALSE;
	pPass->GetAnnotationByName("RevertTargets")->AsScalar()->GetBool(&bRevertTargets);
	if (bRevertTargets != FALSE)
		revertMask |= DX11::StateMasks::RenderTargets;

	return revertMask;
}

/// Extracts all resource bindings from the given pass.
StateEffectBinderPass::ResourceBindings::binding_vector GetResourceBindings(Any::API::Effect *pEffect, Any::API::EffectPass *pPass,
	const char *annotationName, StateEffectBinderPass::ShaderStageStateMask &stateMask)
{
	StateEffectBinderPass::ResourceBindings::binding_vector resourceBindings;

	Any::API::EffectString *pStringArray = pPass->GetAnnotationByName(annotationName)->AsString();
	
	if (pStringArray->IsValid())
	{
		const uint4 stringCount = GetDesc(pStringArray->GetType()).Elements;

		for (uint4 i = 0; i < stringCount; ++i)
		{
			const char *resourceName = "";
			pStringArray->GetStringArray(&resourceName, i, 1);

			// Constant buffers are treated separately
			Any::API::EffectConstantBuffer *pConstants = pEffect->GetConstantBufferByName(resourceName);

			if (!pConstants->IsValid())
				pConstants = nullptr;

			// Everything else is a shader variable
			Any::API::EffectVariable *pVariable = (pConstants)
				? pConstants
				: pEffect->GetVariableByName(resourceName);

			if (pVariable->IsValid())
			{
				D3DX11_EFFECT_VARIABLE_DESC variableDesc = GetDesc(pVariable);

				// Manual resource binding only makes sense for resources bound to fixed registers
				if (variableDesc.Flags & D3DX11_EFFECT_VARIABLE_EXPLICIT_BIND_POINT)
				{
					if (pConstants)
					{
						// Add constant buffer binding
						resourceBindings.push_back( ResourceBinding(pConstants, variableDesc.ExplicitBindPoint) );
						stateMask.constantBufferMask |= 1U << variableDesc.ExplicitBindPoint;
					}
					else
					{
						Any::API::EffectShaderResource *pResource = pVariable->AsShaderResource();

						if (pResource->IsValid())
						{
							// Add resource binding
							resourceBindings.push_back( ResourceBinding(pResource, variableDesc.ExplicitBindPoint) );
							stateMask.resourceMask |= 1U << variableDesc.ExplicitBindPoint;
						}
						else
							LEAN_LOG_ERROR_CTX("Resource binding currently only supported for constant buffers & resource views.", resourceName);
					}
				}
				else
					LEAN_LOG_ERROR_CTX("Manual resource binding requires explicit bind point.", resourceName);
			}
		}
	}
	
	return resourceBindings;
}

/// Extracts all resource bindings from the given pass.
StateEffectBinderPass::ResourceBindings* GetResourceBindings(Any::API::Effect *pEffect, Any::API::EffectPass *pPass,
	StateEffectBinderPass::StateMask &stateMask)
{
	lean::scoped_ptr<StateEffectBinderPass::ResourceBindings> resourceBindings( new StateEffectBinderPass::ResourceBindings() );

	resourceBindings->VSResourceBindings = GetResourceBindings(pEffect, pPass, "VSBindResources", stateMask.VSMask);
	resourceBindings->HSResourceBindings = GetResourceBindings(pEffect, pPass, "HSBindResources", stateMask.HSMask);
	resourceBindings->DSResourceBindings = GetResourceBindings(pEffect, pPass, "DSBindResources", stateMask.DSMask);
	resourceBindings->GSResourceBindings = GetResourceBindings(pEffect, pPass, "GSBindResources", stateMask.GSMask);
	resourceBindings->PSResourceBindings = GetResourceBindings(pEffect, pPass, "PSBindResources", stateMask.PSMask);
	resourceBindings->CSResourceBindings = GetResourceBindings(pEffect, pPass, "CSBindResources", stateMask.CSMask);

	bool bHasResourceBindings =
		   !resourceBindings->VSResourceBindings.empty()
		|| !resourceBindings->HSResourceBindings.empty()
		|| !resourceBindings->DSResourceBindings.empty()
		|| !resourceBindings->GSResourceBindings.empty()
		|| !resourceBindings->PSResourceBindings.empty()
		|| !resourceBindings->CSResourceBindings.empty();

	return (bHasResourceBindings) ? resourceBindings.detach() : nullptr;
}

/// Gets the number of control points.
uint4 GetControlPointCount(Any::API::EffectPass *pPass)
{
	int controlPoints = 0;
	pPass->GetAnnotationByName("HSControlPoints")->AsScalar()->GetInt(&controlPoints);
	return static_cast<uint4>(controlPoints);
}

} // namespace

// Constructor.
StateEffectBinderPass::StateEffectBinderPass(Any::API::Effect *pEffect, Any::API::EffectPass *pPass, uint4 passID)
	: m_pPass( LEAN_ASSERT_NOT_NULL(pPass) ),
	m_passID(passID),
	m_stateMask( GetStateMask(pPass) ),
	m_pipelineRevertMask( GetStateRevertMask(pPass) ),
	m_pResourceBindings( GetResourceBindings(pEffect, pPass, m_stateMask) ),
	m_controlPointCount( GetControlPointCount(pPass) )
{
}

// Destructor.
StateEffectBinderPass::~StateEffectBinderPass()
{
}

namespace
{

/// Updates the given state manager with the given shader state.
template <class StateManager>
void UpdateShaderState(StateManager &shaderStateManager, const StateEffectBinderPass::ShaderStageStateMask &stateMask)
{
	shaderStateManager.OverrideShader(stateMask.shaderSet);
	shaderStateManager.OverrideConstantBuffers(stateMask.constantBufferMask);
	shaderStateManager.OverrideResources(stateMask.resourceMask);
}

/// Applies the given resource bindings.
template <
		void (STDMETHODCALLTYPE ID3D11DeviceContext::*SetShaderResources)(UINT, UINT, ID3D11ShaderResourceView*const*),
		void (STDMETHODCALLTYPE ID3D11DeviceContext::*SetConstantBuffers)(UINT, UINT, ID3D11Buffer*const*),
		class Vector
	>
void ApplyResourceBindings(const Vector &bindings, Any::API::DeviceContext *pContext)
{
	for (typename Vector::const_iterator it = bindings.begin(); it != bindings.end(); ++it)
		switch (it->type)
		{
		case ResourceBindingType::Resource:
			{
				lean::com_ptr<ID3D11ShaderResourceView> pResource;
				static_cast<Any::API::EffectShaderResource*>(it->pResource)->GetResource(pResource.rebind());
				(pContext->*SetShaderResources)( it->bindPoint, 1, &pResource.get() );
			}
			break;

		case ResourceBindingType::ConstantBuffer:
			{
				lean::com_ptr<ID3D11Buffer> pBuffer;
				static_cast<Any::API::EffectConstantBuffer*>(it->pResource)->GetConstantBuffer(pBuffer.rebind());
				(pContext->*SetConstantBuffers)( it->bindPoint, 1, &pBuffer.get() );
			}
			break;
		}
}

} // namespace

// Applies the pass the n-th time.
bool StateEffectBinderPass::Apply(Any::StateManager& stateManager, Any::API::DeviceContext *pContext) const
{
	LEAN_ASSERT( (m_pipelineRevertMask & DX11::StateMasks::PipelineStates) == m_pipelineRevertMask );
	stateManager.Revert(m_pipelineRevertMask);

	LEAN_ASSERT( (m_stateMask.pipelineMask & DX11::StateMasks::PipelineStates) == m_stateMask.pipelineMask );
	stateManager.Override(m_stateMask.pipelineMask);

	UpdateShaderState(stateManager.VS(), m_stateMask.VSMask);
	UpdateShaderState(stateManager.HS(), m_stateMask.HSMask);
	UpdateShaderState(stateManager.DS(), m_stateMask.DSMask);
	UpdateShaderState(stateManager.GS(), m_stateMask.GSMask);
	UpdateShaderState(stateManager.PS(), m_stateMask.PSMask);
	UpdateShaderState(stateManager.CS(), m_stateMask.CSMask);

	if (m_pResourceBindings)
	{
		ApplyResourceBindings<&ID3D11DeviceContext::VSSetShaderResources, &ID3D11DeviceContext::VSSetConstantBuffers>(m_pResourceBindings->VSResourceBindings, pContext);
		ApplyResourceBindings<&ID3D11DeviceContext::HSSetShaderResources, &ID3D11DeviceContext::HSSetConstantBuffers>(m_pResourceBindings->HSResourceBindings, pContext);
		ApplyResourceBindings<&ID3D11DeviceContext::DSSetShaderResources, &ID3D11DeviceContext::DSSetConstantBuffers>(m_pResourceBindings->DSResourceBindings, pContext);
		ApplyResourceBindings<&ID3D11DeviceContext::GSSetShaderResources, &ID3D11DeviceContext::GSSetConstantBuffers>(m_pResourceBindings->GSResourceBindings, pContext);
		ApplyResourceBindings<&ID3D11DeviceContext::PSSetShaderResources, &ID3D11DeviceContext::PSSetConstantBuffers>(m_pResourceBindings->PSResourceBindings, pContext);
		ApplyResourceBindings<&ID3D11DeviceContext::CSSetShaderResources, &ID3D11DeviceContext::CSSetConstantBuffers>(m_pResourceBindings->CSResourceBindings, pContext);
	}

	HRESULT result = m_pPass->Apply(0, pContext);
#ifdef LEAN_DEBUG_BUILD
	BE_LOG_DX_ERROR_MSG(result, "ID3DX11EffectPass::Apply()");
#endif

	if (m_controlPointCount > 0)
		pContext->IASetPrimitiveTopology(
				static_cast<D3D11_PRIMITIVE_TOPOLOGY>(D3D11_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST + m_controlPointCount - 1)
			);
	
	return true;
}

// Gets the input signature of this pass.
const char* StateEffectBinderPass::GetInputSignature(uint4 &size) const
{
	D3DX11_PASS_DESC passDesc;
	BE_THROW_DX_ERROR_MSG(m_pPass->GetDesc(&passDesc), "ID3DX11EffectPass::GetDesc()");

	size = static_cast<uint4>(passDesc.IAInputSignatureSize);
	return reinterpret_cast<const char*>(passDesc.pIAInputSignature);
}

namespace
{

// Gets all passes in the given effect.
StateEffectBinder::pass_vector GetPasses(Any::API::Effect *pEffect, Any::API::EffectTechnique *pTechnique)
{
	StateEffectBinder::pass_vector passes;

	D3DX11_TECHNIQUE_DESC techDesc;
	BE_THROW_DX_ERROR_MSG(
		pTechnique->GetDesc(&techDesc),
		"ID3DX11EffectTechnique::GetDesc()");

	passes.reset(techDesc.Passes);

	for (UINT passID = 0; passID < techDesc.Passes; ++passID)
	{
		Any::API::EffectPass *pPass = pTechnique->GetPassByIndex(passID);

		if (!pPass->IsValid())
			LEAN_THROW_ERROR_MSG("Invalid pass");

		new_emplace(passes) StateEffectBinderPass(pEffect, pPass, passID);
	}

	return passes;
}

} // namespace

// Constructor.
StateEffectBinder::StateEffectBinder(const Any::Technique &technique)
	: m_technique( technique ),
	m_passes( beScene::GetPasses(*m_technique.GetEffect(), m_technique), lean::consume )
{
}

// Destructor.
StateEffectBinder::~StateEffectBinder()
{
}

// Gets the passes.
StateEffectBinder::PassRange StateEffectBinder::GetPasses() const
{
	return beCore::MakeRangeN(&m_passes[0], m_passes.size());
}

} // namespace