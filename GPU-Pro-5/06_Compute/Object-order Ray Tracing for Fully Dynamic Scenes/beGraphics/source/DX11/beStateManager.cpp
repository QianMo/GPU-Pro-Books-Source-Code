/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#include "beGraphicsInternal/stdafx.h"

#include "beGraphics/DX11/beStateManager.h"
#include "beGraphics/DX11/beTexture.h"
#include "beGraphics/DX/beError.h"
#include <lean/functional/algorithm.h>

namespace beGraphics
{

namespace DX11
{

// Constructor.
StateManager::StateManager(ID3D11DeviceContext *pContext)
	: m_pContext(pContext),

	m_VSManager(this, &m_setup.VSState),
	m_HSManager(this, &m_setup.HSState),
	m_DSManager(this, &m_setup.DSState),
	m_GSManager(this, &m_setup.GSState),
	m_PSManager(this, &m_setup.PSState),
	m_CSManager(this, &m_setup.CSState),
	
	m_invalidMask(0),
	m_overrideMask(0)
{
}

// Destructor.
StateManager::~StateManager()
{
}

// Records overridden state.
void StateManager::RecordOverridden()
{
	DX11::Record(m_setup, m_pContext, m_overrideMask);
	
	if (m_overrideMask & StateMasks::VertexShader)
		if (m_VSManager.RecordOverridden())
			m_setup.StateMask |= StateMasks::VertexShader;
	if (m_overrideMask & StateMasks::HullShader)
		if (m_HSManager.RecordOverridden())
			m_setup.StateMask |= StateMasks::HullShader;
	if (m_overrideMask & StateMasks::DomainShader)
		if (m_DSManager.RecordOverridden())
			m_setup.StateMask |= StateMasks::DomainShader;
	if (m_overrideMask & StateMasks::GeometryShader)
		if (m_GSManager.RecordOverridden())
			m_setup.StateMask |= StateMasks::GeometryShader;
	if (m_overrideMask & StateMasks::PixelShader)
		if (m_PSManager.RecordOverridden())
			m_setup.StateMask |= StateMasks::PixelShader;
	if (m_overrideMask & StateMasks::ComputeShader)
		if (m_CSManager.RecordOverridden())
			m_setup.StateMask |= StateMasks::ComputeShader;

	Revert(m_setup.StateMask);
}

// Sets the given states.
void StateManager::Set(const beGraphics::StateSetup& setup)
{
	m_setup = ToImpl(setup);

	Invalidate();
	m_VSManager.Invalidate();
	m_HSManager.Invalidate();
	m_DSManager.Invalidate();
	m_GSManager.Invalidate();
	m_PSManager.Invalidate();
	m_CSManager.Invalidate();
}

// Clears the given states.
void StateManager::Clear(uint4 stateMask)
{
	DX11::Clear(m_setup, stateMask);
}

// Records the given states.
void RecordConstantBuffers(ShaderStageStateSetup &setup, ID3D11DeviceContext *pContext,
	void (_stdcall ID3D11DeviceContext::*getConstantBuffers)(UINT, UINT, ID3D11Buffer**),
	uint4 bufferMask)
{
	ID3D11Buffer *buffers[ShaderStageStateSetup::MaxConstantBuffers] = { nullptr };
	(pContext->*getConstantBuffers)(0, ShaderStageStateSetup::MaxConstantBuffers, buffers);

	// WARNING: Always clamp mask to valid range
	bufferMask &= ShaderStageStateSetup::AllConstantBuffers;
	setup.ConstantBufferMask = bufferMask;

	for (size_t i = 0; i < ShaderStageStateSetup::MaxConstantBuffers; ++i, bufferMask >>= 1)
	{
		// WARNING: Always bind ALL retrieved pointers ...
		setup.ConstantBuffers[i] = lean::bind_com(buffers[i]);

		/// ... before selectively RELEASING some of them
		if (~bufferMask & 0x1)
			setup.ConstantBuffers[i] = nullptr;
	}
}

// Records the given states.
void RecordResources(ShaderStageStateSetup &setup, ID3D11DeviceContext *pContext,
	void (_stdcall ID3D11DeviceContext::*getResources)(UINT, UINT, ID3D11ShaderResourceView**),
	uint4 resourceMask)
{
	ID3D11ShaderResourceView *resources[ShaderStageStateSetup::MaxResources] = { nullptr };
	(pContext->*getResources)(0, ShaderStageStateSetup::MaxResources, resources);

	// WARNING: Always clamp mask to valid range
	resourceMask &= ShaderStageStateSetup::AllResources;
	setup.ResourceMask = resourceMask;

	for (size_t i = 0; i < ShaderStageStateSetup::MaxResources; ++i, resourceMask >>= 1)
	{
		// WARNING: Always bind ALL retrieved pointers ...
		setup.Resources[i] = lean::bind_com(resources[i]);

		/// ... before selectively RELEASING some of them
		if (~resourceMask & 0x1)
			setup.Resources[i] = nullptr;
	}
}

// Records the given states.
void Record(ShaderStageStateSetup &setup, ID3D11DeviceContext *pContext,
	void (_stdcall ID3D11DeviceContext::*getConstantBuffers)(UINT, UINT, ID3D11Buffer**),
	void (_stdcall ID3D11DeviceContext::*getResources)(UINT, UINT, ID3D11ShaderResourceView**),
	uint4 bufferMask,
	uint4 resourceMask)
{
	RecordConstantBuffers(setup, pContext, getConstantBuffers, bufferMask);
	RecordResources(setup, pContext, getResources, resourceMask);
}

// Records the given states.
void Record(StateSetup &setup, ID3D11DeviceContext *pContext, uint4 stateMask)
{
	if (stateMask & StateMasks::VertexShader)
	{
		Record(setup.VSState, pContext, &ID3D11DeviceContext::VSGetConstantBuffers, &ID3D11DeviceContext::VSGetShaderResources);
		setup.StateMask |= StateMasks::VertexShader;
	}
	if (stateMask & StateMasks::HullShader)
	{
		Record(setup.HSState, pContext, &ID3D11DeviceContext::HSGetConstantBuffers, &ID3D11DeviceContext::HSGetShaderResources);
		setup.StateMask |= StateMasks::HullShader;
	}
	if (stateMask & StateMasks::DomainShader)
	{
		Record(setup.DSState, pContext, &ID3D11DeviceContext::DSGetConstantBuffers, &ID3D11DeviceContext::DSGetShaderResources);
		setup.StateMask |= StateMasks::DomainShader;
	}
	if (stateMask & StateMasks::GeometryShader)
	{
		Record(setup.GSState, pContext, &ID3D11DeviceContext::GSGetConstantBuffers, &ID3D11DeviceContext::GSGetShaderResources);
		setup.StateMask |= StateMasks::GeometryShader;
	}
	if (stateMask & StateMasks::PixelShader)
	{
		Record(setup.PSState, pContext, &ID3D11DeviceContext::PSGetConstantBuffers, &ID3D11DeviceContext::PSGetShaderResources);
		setup.StateMask |= StateMasks::PixelShader;
	}
	if (stateMask & StateMasks::ComputeShader)
	{
		Record(setup.CSState, pContext, &ID3D11DeviceContext::CSGetConstantBuffers, &ID3D11DeviceContext::CSGetShaderResources);
		setup.StateMask |= StateMasks::ComputeShader;
	}

	if (stateMask & StateMasks::RasterizerState)
	{
		pContext->RSGetState(setup.RasterizerState.rebind());
		setup.StateMask |= StateMasks::RasterizerState;
	}
	if (stateMask & StateMasks::DepthStencilState)
	{
		pContext->OMGetDepthStencilState(setup.DepthStencilState.rebind(), &setup.StencilRef);
		setup.StateMask |= StateMasks::DepthStencilState;
	}
	if (stateMask & StateMasks::BlendState)
	{
		pContext->OMGetBlendState(setup.BlendState.rebind(), setup.BlendFactor, &setup.SampleMask);
		setup.StateMask |= StateMasks::BlendState;
	}
	if (stateMask & StateMasks::RenderTargets)
	{
		ID3D11RenderTargetView *renderTargets[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = { nullptr };
		pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, renderTargets, setup.DepthStencilTarget.rebind());

		UINT maxTargetIndex = -1;

		for (UINT i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
		{
			if (renderTargets[i])	
				maxTargetIndex = i;

			// WARNING: Always bind ALL retrieved pointers
			setup.RenderTargets[i] = lean::bind_com(renderTargets[i]);
		}
		setup.RenderTargetCount = maxTargetIndex + 1;

		setup.StateMask |= StateMasks::RenderTargets;
	}
}

// Clears the given shader stage states.
void ClearConstantBuffers(ShaderStageStateSetup &setup, uint4 bufferMask)
{
	setup.ConstantBufferMask &= ~bufferMask;

	// WARNING: Always clamp mask to valid range
	bufferMask &= ShaderStageStateSetup::MaxConstantBuffers;
	
	for (size_t i = 0; bufferMask != 0; ++i, bufferMask >>= 1)
		if (bufferMask & 0x1)
			setup.ConstantBuffers[i] = nullptr;
}

// Clears the given shader stage states.
void ClearResources(ShaderStageStateSetup &setup, uint4 resourceMask)
{
	setup.ResourceMask &= ~resourceMask;

	// WARNING: Always clamp mask to valid range
	resourceMask &= ShaderStageStateSetup::MaxResources;
	
	for (size_t i = 0; resourceMask != 0; ++i, resourceMask >>= 1)
		if (resourceMask & 0x1)
			setup.Resources[i] = nullptr;
}

// Clears the given shader stage states.
void Clear(ShaderStageStateSetup &setup,
	uint4 bufferMask,
	uint4 resourceMask)
{
	ClearConstantBuffers(setup, bufferMask);
	ClearResources(setup, resourceMask);
}

// Clears the given states.
void Clear(StateSetup &setup, uint4 stateMask)
{
	setup.StateMask &= ~stateMask;

	if (stateMask & StateMasks::VertexShader)
		Clear(setup.VSState);
	if (stateMask & StateMasks::HullShader)
		Clear(setup.HSState);
	if (stateMask & StateMasks::DomainShader)
		Clear(setup.DSState);
	if (stateMask & StateMasks::GeometryShader)
		Clear(setup.GSState);
	if (stateMask & StateMasks::PixelShader)
		Clear(setup.PSState);
	if (stateMask & StateMasks::ComputeShader)
		Clear(setup.CSState);

	if (stateMask & StateMasks::RasterizerState)
		setup.RasterizerState = nullptr;
	if (stateMask & StateMasks::DepthStencilState)
		setup.DepthStencilState = nullptr;
	if (stateMask & StateMasks::BlendState)
		setup.BlendState = nullptr;
	if (stateMask & StateMasks::RenderTargets)
	{
		for (UINT i = 0; i < setup.RenderTargetCount; ++i)
			setup.RenderTargets[i] = nullptr;
		setup.RenderTargetCount = 0;
		setup.DepthStencilTarget = nullptr;
	}
}

} // namespace

} // namespace
