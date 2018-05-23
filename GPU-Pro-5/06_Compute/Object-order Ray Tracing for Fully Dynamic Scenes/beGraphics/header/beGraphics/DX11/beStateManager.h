/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_STATE_MANAGER_DX11
#define BE_GRAPHICS_STATE_MANAGER_DX11

#include "beGraphics.h"
#include "../beStateManager.h"
#include <D3D11.h>
#include <lean/smart/com_ptr.h>
#include <vector>

namespace beGraphics
{

namespace DX11
{

/// State mask enumeration.
struct StateMasks
{
	/// Enumeration.
	enum T
	{
		RasterizerState = 1 << 0,		///< Rasterizer state mask.
		DepthStencilState = 1 << 1,		///< Depth-stencil state mask.
		BlendState = 1 << 2,			///< Blend state mask.
		RenderTargets = 1 << 3,			///< Render target state mask.

		PipelineStates = RasterizerState | DepthStencilState | BlendState | RenderTargets,

		VertexShader = 1 << 4,			///< Vertex shader state mask.
		HullShader = 1 << 5,			///< Hull shader state mask.
		DomainShader = 1 << 6,			///< Domain shader state mask.
		GeometryShader = 1 << 7,		///< Geometry shader state mask.
		PixelShader = 1 << 8,			///< Pixel shader state mask.
		ComputeShader = 1 << 9,			///< Compute shader state mask.

		ShaderStates = VertexShader | HullShader | DomainShader | GeometryShader | PixelShader | ComputeShader,

		AllStates = PipelineStates | ShaderStates
	};
	LEAN_MAKE_ENUM_STRUCT(StateMasks)
};

/// Shader stage state setup.
struct ShaderStageStateSetup
{
	bool ShaderSet;
	lean::com_ptr<ID3D11DeviceChild> Shader;

	static const size_t MaxConstantBuffers = (D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT < lean::size_info<uint4>::bits)
		? D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT
		: lean::size_info<uint4>::bits;
	static const uint4 AllConstantBuffers = static_cast<uint4>(-1) >> (lean::size_info<uint4>::bits - MaxConstantBuffers);

	uint4 ConstantBufferMask;											///< Valid constant buffers.
	lean::com_ptr<ID3D11Buffer> ConstantBuffers[MaxConstantBuffers];	///< Constant buffers.

	static const size_t MaxResources = (D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT < lean::size_info<uint4>::bits)
		? D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT
		: lean::size_info<uint4>::bits;
	static const uint4 AllResources = static_cast<uint4>(-1) >> (lean::size_info<uint4>::bits - MaxResources);

	uint4 ResourceMask;													///< Valid shader resources.
	lean::com_ptr<ID3D11ShaderResourceView> Resources[MaxResources];	///< Shader Resources.
};

/// Records the given states.
BE_GRAPHICS_DX11_API void RecordConstantBuffers(ShaderStageStateSetup &setup, ID3D11DeviceContext *pContext,
	void (_stdcall ID3D11DeviceContext::*getConstantBuffers)(UINT, UINT, ID3D11Buffer**),
	uint4 bufferMask = ShaderStageStateSetup::AllConstantBuffers);
/// Records the given states.
BE_GRAPHICS_DX11_API void RecordResources(ShaderStageStateSetup &setup, ID3D11DeviceContext *pContext,
	void (_stdcall ID3D11DeviceContext::*getResources)(UINT, UINT, ID3D11ShaderResourceView**),
	uint4 resourceMask = ShaderStageStateSetup::AllResources);
/// Records the given states.
BE_GRAPHICS_DX11_API void Record(ShaderStageStateSetup &setup, ID3D11DeviceContext *pContext,
	void (_stdcall ID3D11DeviceContext::*getConstantBuffers)(UINT, UINT, ID3D11Buffer**),
	void (_stdcall ID3D11DeviceContext::*getResources)(UINT, UINT, ID3D11ShaderResourceView**),
	uint4 bufferMask = ShaderStageStateSetup::AllConstantBuffers,
	uint4 resourceMask = ShaderStageStateSetup::AllResources);

/// Clears the given shader stage states.
BE_GRAPHICS_DX11_API void ClearConstantBuffers(ShaderStageStateSetup &setup, uint4 bufferMask = ShaderStageStateSetup::AllConstantBuffers);
/// Clears the given shader stage states.
BE_GRAPHICS_DX11_API void ClearResources(ShaderStageStateSetup &setup, uint4 resourceMask = ShaderStageStateSetup::AllResources);
/// Clears the given shader stage states.
BE_GRAPHICS_DX11_API void Clear(ShaderStageStateSetup &setup,
	uint4 bufferMask = ShaderStageStateSetup::AllConstantBuffers,
	uint4 resourceMask = ShaderStageStateSetup::AllResources);

/// State setup implementation.
struct StateSetup : public beGraphics::StateSetup
{
	uint4 StateMask;				///< Valid state mask.

	ShaderStageStateSetup VSState;	///< Vertex shader stage state.
	ShaderStageStateSetup HSState;	///< Hull shader stage state.
	ShaderStageStateSetup DSState;	///< Domain shader stage state.
	ShaderStageStateSetup GSState;	///< Geometry shader stage state.
	ShaderStageStateSetup PSState;	///< Pixel shader stage state.
	ShaderStageStateSetup CSState;	///< Compute shader stage state.

	lean::com_ptr<ID3D11RasterizerState> RasterizerState;		///< Rasterizer state.

	lean::com_ptr<ID3D11DepthStencilState> DepthStencilState;	///< Depth-stencil state.
	UINT StencilRef;											///< Stencil reference value.
	
	lean::com_ptr<ID3D11BlendState> BlendState;					///< Blend state.
	FLOAT BlendFactor[4];										///< Blend factors.
	UINT SampleMask;											///< Sample mask.

	lean::com_ptr<ID3D11RenderTargetView> RenderTargets[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];	///< Render targets.
	UINT RenderTargetCount;																			///< Valid render target count.
	lean::com_ptr<ID3D11DepthStencilView> DepthStencilTarget;										///< Depth-stencil target.

	/// Constructor.
	StateSetup(uint4 mask = 0)
		: StateMask(mask),
		VSState(),
		HSState(),
		DSState(),
		GSState(),
		PSState(),
		CSState(),
		RenderTargetCount(0) { }

	/// Gets the implementation identifier.
	ImplementationID GetImplementationID() const { return DX11Implementation; };
};

/// Records the given states.
BE_GRAPHICS_DX11_API void Record(StateSetup &setup, ID3D11DeviceContext *pContext, uint4 stateMask = StateMasks::AllStates);
/// Clears the given states.
BE_GRAPHICS_DX11_API void Clear(StateSetup &setup, uint4 stateMask = StateMasks::AllStates);

template <> struct ToImplementationDX11<beGraphics::StateSetup> { typedef StateSetup Type; };

// Prototypes
class StateManager;

/// Shader stage traits.
namespace ShaderStageStateTraits
{

template <class ID3D11ShaderInterface>
void GetShader(ID3D11DeviceContext *pContext, ID3D11DeviceChild **ppShader);
template <class ID3D11ShaderInterface>
void SetShader(ID3D11DeviceContext *pContext, ID3D11DeviceChild *pShader);

template <>
LEAN_INLINE void SetShader<ID3D11VertexShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild *pShader)
{
	pContext->VSSetShader(static_cast<ID3D11VertexShader*>(pShader), nullptr, 0);
}
template <>
LEAN_INLINE void GetShader<ID3D11VertexShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild **ppShader)
{
	ID3D11VertexShader *pShader;
	UINT numIfcs = 0;
	pContext->VSGetShader(&pShader, nullptr, &numIfcs);
	*ppShader = pShader;
}

template <>
LEAN_INLINE void SetShader<ID3D11HullShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild *pShader)
{
	pContext->HSSetShader(static_cast<ID3D11HullShader*>(pShader), nullptr, 0);
}
template <>
LEAN_INLINE void GetShader<ID3D11HullShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild **ppShader)
{
	ID3D11HullShader *pShader;
	UINT numIfcs = 0;
	pContext->HSGetShader(&pShader, nullptr, &numIfcs);
	*ppShader = pShader;
}

template <>
LEAN_INLINE void SetShader<ID3D11DomainShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild *pShader)
{
	pContext->DSSetShader(static_cast<ID3D11DomainShader*>(pShader), nullptr, 0);
}
template <>
LEAN_INLINE void GetShader<ID3D11DomainShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild **ppShader)
{
	ID3D11DomainShader *pShader;
	UINT numIfcs = 0;
	pContext->DSGetShader(&pShader, nullptr, &numIfcs);
	*ppShader = pShader;
}

template <>
LEAN_INLINE void SetShader<ID3D11GeometryShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild *pShader)
{
	pContext->GSSetShader(static_cast<ID3D11GeometryShader*>(pShader), nullptr, 0);
}
template <>
LEAN_INLINE void GetShader<ID3D11GeometryShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild **ppShader)
{
	ID3D11GeometryShader *pShader;
	UINT numIfcs = 0;
	pContext->GSGetShader(&pShader, nullptr, &numIfcs);
	*ppShader = pShader;
}

template <>
LEAN_INLINE void SetShader<ID3D11PixelShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild *pShader)
{
	pContext->PSSetShader(static_cast<ID3D11PixelShader*>(pShader), nullptr, 0);
}
template <>
LEAN_INLINE void GetShader<ID3D11PixelShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild **ppShader)
{
	ID3D11PixelShader *pShader;
	UINT numIfcs = 0;
	pContext->PSGetShader(&pShader, nullptr, &numIfcs);
	*ppShader = pShader;
}

template <>
LEAN_INLINE void SetShader<ID3D11ComputeShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild *pShader)
{
	pContext->CSSetShader(static_cast<ID3D11ComputeShader*>(pShader), nullptr, 0);
}
template <>
LEAN_INLINE void GetShader<ID3D11ComputeShader>(ID3D11DeviceContext *pContext, ID3D11DeviceChild **ppShader)
{
	ID3D11ComputeShader *pShader;
	UINT numIfcs = 0;
	pContext->CSGetShader(&pShader, nullptr, &numIfcs);
	*ppShader = pShader;
}

/// Shader stage traits.
template <uint4 StateMask, ShaderStageStateSetup (StateSetup::*StageStateSetup),
	class ID3D11ShaderInterface,
	void (_stdcall ID3D11DeviceContext::*SetConstantBuffersP)(UINT, UINT, ID3D11Buffer *const *),
	void (_stdcall ID3D11DeviceContext::*GetConstantBuffersP)(UINT, UINT, ID3D11Buffer**),
	void (_stdcall ID3D11DeviceContext::*SetShaderResourcesP)(UINT, UINT, ID3D11ShaderResourceView *const *),
	void (_stdcall ID3D11DeviceContext::*GetShaderResourcesP)(UINT, UINT, ID3D11ShaderResourceView**)>
struct Traits
{
	/// State mask.
	static const uint4 StateMask = StateMask;

	/// Shader interface.
	typedef ID3D11ShaderInterface ID3D11ShaderInterface;

	/// Gets the shader state state setup from the given state setup.
	static LEAN_INLINE ShaderStageStateSetup& GetSetup(StateSetup &setup) { return (setup.*StageStateSetup); }
	/// Gets the shader state state setup from the given state setup.
	static LEAN_INLINE const ShaderStageStateSetup& GetSetup(const StateSetup &setup) { return (setup.*StageStateSetup); }

	/// @code SetConstantBuffers()@endcode method pointer.
	static LEAN_INLINE void (_stdcall ID3D11DeviceContext::* SetConstantBuffersPtr() )(UINT, UINT, ID3D11Buffer *const *) { return SetConstantBuffersP; }
	/// @code GetConstantBuffers()@endcode method pointer.
	static LEAN_INLINE void (_stdcall ID3D11DeviceContext::* GetConstantBuffersPtr() )(UINT, UINT, ID3D11Buffer**) { return GetConstantBuffersP; }

	/// Sets the given shader in the given device context.
	static LEAN_INLINE void SetShader(ID3D11DeviceContext *pContext, ID3D11DeviceChild *pShader)
	{
		ShaderStageStateTraits::SetShader<ID3D11ShaderInterface>(pContext, pShader);
	}
	/// Gets the shader from the given device context.
	static LEAN_INLINE void GetShader(ID3D11DeviceContext *pContext, ID3D11DeviceChild **ppShader)
	{
		ShaderStageStateTraits::GetShader<ID3D11ShaderInterface>(pContext, ppShader);
	}

	/// Sets constant buffers in the given device context.
	static LEAN_INLINE void SetConstantBuffers(ID3D11DeviceContext *pContext, UINT offset, UINT count, ID3D11Buffer *const *buffers)
	{
		(pContext->*SetConstantBuffersP)(offset, count, buffers);
	}
	/// Gets constant buffers from the given device context.
	static LEAN_INLINE void GetConstantBuffers(ID3D11DeviceContext *pContext, UINT offset, UINT count, ID3D11Buffer **buffers)
	{
		(pContext->*GetConstantBuffersP)(offset, count, buffers);
	}

	/// @code SetShaderResources()@endcode method pointer.
	static LEAN_INLINE void (_stdcall ID3D11DeviceContext::* SetShaderResourcesPtr() )(UINT, UINT, ID3D11ShaderResourceView *const *) { return SetShaderResourcesP; }
	/// @code GetShaderResources()@endcode method pointer.
	static LEAN_INLINE void (_stdcall ID3D11DeviceContext::* GetShaderResourcesPtr() )(UINT, UINT, ID3D11ShaderResourceView**) { return GetShaderResourcesP; }

	/// Sets resources in the given device context.
	static LEAN_INLINE void SetShaderResources(ID3D11DeviceContext *pContext, UINT offset, UINT count, ID3D11ShaderResourceView *const *resources)
	{
		(pContext->*SetShaderResourcesP)(offset, count, resources);
	}
	/// Gets resources from the given device context.
	static LEAN_INLINE void GetShaderResources(ID3D11DeviceContext *pContext, UINT offset, UINT count, ID3D11ShaderResourceView **resources)
	{
		(pContext->*GetShaderResourcesP)(offset, count, resources);
	}
};

typedef Traits<StateMasks::VertexShader, &StateSetup::VSState,
	ID3D11VertexShader,
	&ID3D11DeviceContext::VSSetConstantBuffers, &ID3D11DeviceContext::VSGetConstantBuffers, 
	&ID3D11DeviceContext::VSSetShaderResources, &ID3D11DeviceContext::VSGetShaderResources> VertexShader;
typedef Traits<StateMasks::HullShader, &StateSetup::HSState,
	ID3D11HullShader,
	&ID3D11DeviceContext::HSSetConstantBuffers, &ID3D11DeviceContext::HSGetConstantBuffers, 
	&ID3D11DeviceContext::HSSetShaderResources, &ID3D11DeviceContext::HSGetShaderResources> HullShader;
typedef Traits<StateMasks::DomainShader, &StateSetup::DSState,
	ID3D11DomainShader,
	&ID3D11DeviceContext::DSSetConstantBuffers, &ID3D11DeviceContext::DSGetConstantBuffers, 
	&ID3D11DeviceContext::DSSetShaderResources, &ID3D11DeviceContext::DSGetShaderResources> DomainShader;
typedef Traits<StateMasks::GeometryShader, &StateSetup::GSState,
	ID3D11GeometryShader,
	&ID3D11DeviceContext::GSSetConstantBuffers, &ID3D11DeviceContext::GSGetConstantBuffers, 
	&ID3D11DeviceContext::GSSetShaderResources, &ID3D11DeviceContext::GSGetShaderResources> GeometryShader;
typedef Traits<StateMasks::PixelShader, &StateSetup::PSState,
	ID3D11PixelShader,
	&ID3D11DeviceContext::PSSetConstantBuffers, &ID3D11DeviceContext::PSGetConstantBuffers, 
	&ID3D11DeviceContext::PSSetShaderResources, &ID3D11DeviceContext::PSGetShaderResources> PixelShader;
typedef Traits<StateMasks::ComputeShader, &StateSetup::CSState,
	ID3D11ComputeShader,
	&ID3D11DeviceContext::CSSetConstantBuffers, &ID3D11DeviceContext::CSGetConstantBuffers, 
	&ID3D11DeviceContext::CSSetShaderResources, &ID3D11DeviceContext::CSGetShaderResources> ComputeShader;

} // namespace

/// Shader stage state manager.
template <class ShaderStageTraits>
class ShaderStageStateManager
{
private:
	StateManager *m_pStateManager;
	ShaderStageStateSetup *m_pSetup;

	bool m_shaderInvalid;
	bool m_shaderOverride;

	uint4 m_constantBufferInvalidMask;
	uint4 m_constantBufferOverrideMask;

	uint4 m_shaderResourceInvalidMask;
	uint4 m_shaderResourceOverrideMask;

	/// Applies the stored constant buffer state.
	LEAN_INLINE void ApplyConstantBuffer(uint4 index) const
	{
		ShaderStageTraits::SetConstantBuffers(m_pStateManager->GetContext(), index, 1, &m_pSetup->ConstantBuffers[index].get());
	}
	/// Applies the stored shader resource state.
	LEAN_INLINE void ApplyShaderResource(uint4 index) const
	{
		ShaderStageTraits::SetShaderResources(m_pStateManager->GetContext(), index, 1, &m_pSetup->Resources[index].get());
	}

public:
	/// Constructor.
	ShaderStageStateManager(StateManager *pStateManager,
		ShaderStageStateSetup *pSetup)
			: m_pStateManager(pStateManager),
			m_pSetup(pSetup),
			m_shaderInvalid(),
			m_shaderOverride(),
			m_constantBufferInvalidMask(),
			m_constantBufferOverrideMask(),
			m_shaderResourceInvalidMask(),
			m_shaderResourceOverrideMask() { }

	/// Records overridden state.
	bool RecordOverridden()
	{
		m_pSetup->ShaderSet = m_shaderOverride;

		if (m_shaderOverride)
		{
			ShaderStageTraits::GetShader(m_pStateManager->GetContext(), m_pSetup->Shader.rebind());
			m_shaderOverride = false;
		}

		DX11::Record(*m_pSetup, m_pStateManager->GetContext(),
			ShaderStageTraits::GetConstantBuffersPtr(),
			ShaderStageTraits::GetShaderResourcesPtr(),
			m_constantBufferOverrideMask,
			m_shaderResourceOverrideMask);

		m_constantBufferOverrideMask &= ~m_pSetup->ConstantBufferMask;
		m_shaderResourceOverrideMask &= ~m_pSetup->ResourceMask;

		return m_pSetup->ShaderSet | ((m_pSetup->ConstantBufferMask | m_pSetup->ResourceMask) != 0);
	}

	/// Invalidates the given states.
	LEAN_INLINE void InvalidateShader(bool bInvalidate)
	{
		m_shaderInvalid |= bInvalidate;

		if (m_shaderInvalid)
			m_pStateManager->Invalidate(ShaderStageTraits::StateMask);
	}
	/// Overrides the given states.
	LEAN_INLINE void OverrideShader(bool bOverride)
	{
		m_shaderOverride |= bOverride;

		if (m_shaderOverride)
			m_pStateManager->Override(ShaderStageTraits::StateMask);
		
		InvalidateShader(bOverride);
	}
	/// Gets overridden states.
	LEAN_INLINE bool ShaderOverridden() const { return m_shaderOverride; }
	/// Revers the given overridden states.
	LEAN_INLINE void RevertShader(bool bRevert) { m_shaderOverride &= !bRevert; }

	/// Invalidates the given states.
	LEAN_INLINE void InvalidateConstantBuffers(uint4 bufferMask)
	{
		m_constantBufferInvalidMask |= bufferMask;

		if (m_constantBufferInvalidMask)
			m_pStateManager->Invalidate(ShaderStageTraits::StateMask);
	}
	/// Overrides the given states.
	LEAN_INLINE void OverrideConstantBuffers(uint4 bufferMask)
	{
		m_constantBufferOverrideMask |= bufferMask;
		
		if (m_constantBufferOverrideMask)
			m_pStateManager->Override(ShaderStageTraits::StateMask);

		InvalidateConstantBuffers(bufferMask);
	}
	/// Gets overridden states.
	LEAN_INLINE uint4 OverriddenConstantBuffers() const { return m_constantBufferOverrideMask; }
	/// Revers the given overridden states.
	LEAN_INLINE void RevertConstantBuffers(uint4 bufferMask) { m_constantBufferOverrideMask &= ~bufferMask; }

	/// Invalidates the given states.
	LEAN_INLINE void InvalidateResources(uint4 resourceMask)
	{
		m_shaderResourceInvalidMask |= resourceMask;

		if (m_shaderResourceInvalidMask)
			m_pStateManager->Invalidate(ShaderStageTraits::StateMask);
	}
	/// Overrides the given states.
	LEAN_INLINE void OverrideResources(uint4 resourceMask)
	{
		m_shaderResourceOverrideMask |= resourceMask;
		
		if (m_shaderResourceOverrideMask)
			m_pStateManager->Override(ShaderStageTraits::StateMask);

		InvalidateResources(resourceMask);
	}
	/// Gets overridden states.
	LEAN_INLINE uint4 OverriddenResources() const {return m_shaderResourceOverrideMask; }
	/// Reverts the given overridden states.
	LEAN_INLINE void RevertResources(uint4 resourceMask) { m_shaderResourceOverrideMask &= ~resourceMask; }

	/// Invalidates the given states.
	LEAN_INLINE void Invalidate(bool bInvalidateShader = true,
		uint4 constantBufferMask = ShaderStageStateSetup::AllConstantBuffers,
		uint4 resourceMask = ShaderStageStateSetup::AllResources)
	{
		InvalidateShader(bInvalidateShader);
		InvalidateConstantBuffers(constantBufferMask);
		InvalidateResources(resourceMask);
	}
	/// Overrides the given states.
	LEAN_INLINE void Override(bool bOverrideShader = true,
		uint4 constantBufferMask = ShaderStageStateSetup::AllConstantBuffers,
		uint4 resourceMask = ShaderStageStateSetup::AllResources)
	{
		OverrideShader(bOverrideShader);
		OverrideConstantBuffers(constantBufferMask);
		OverrideResources(resourceMask);
	}
	/// Reverts the given overridden states.
	LEAN_INLINE void Revert(bool bRevertShader = true,
		uint4 constantBufferMask = ShaderStageStateSetup::AllConstantBuffers,
		uint4 resourceMask = ShaderStageStateSetup::AllResources)
	{
		RevertShader(bRevertShader);
		RevertConstantBuffers(constantBufferMask);
		RevertResources(resourceMask);
	}

	/// Resets invalid shader states in the given device context.
	void ResetShaderState()
	{
		bool resetShader = m_shaderInvalid & !m_shaderOverride;
		m_shaderInvalid &= !resetShader;

		// Do not reset anything that has not been recorded
		if (resetShader & m_pSetup->ShaderSet)
			ShaderStageTraits::SetShader(m_pStateManager->GetContext(), m_pSetup->Shader);

		uint4 resetConstantBufferMask = m_constantBufferInvalidMask & ~m_constantBufferOverrideMask;
		m_constantBufferInvalidMask &= ~resetConstantBufferMask;

		// Do not reset anything that has not been recorded
		// WARNING: Clamp to valid range
		resetConstantBufferMask &= m_pSetup->ConstantBufferMask & ShaderStageStateSetup::AllConstantBuffers;

		if (resetConstantBufferMask)
			for (uint4 i = 0; resetConstantBufferMask != 0; ++i, resetConstantBufferMask >>= 1)
				if (resetConstantBufferMask & 0x1)
					ApplyConstantBuffer(i);

		uint4 resetShaderResourceMask = m_shaderResourceInvalidMask & ~m_shaderResourceOverrideMask;
		m_shaderResourceInvalidMask &= ~resetShaderResourceMask;

		// Do not reset anything that has not been recorded
		// WARNING: Clamp to valid range
		resetShaderResourceMask &= m_pSetup->ResourceMask & ShaderStageStateSetup::AllResources;

		if (resetShaderResourceMask)
			for (uint4 i = 0; resetShaderResourceMask != 0; ++i, resetShaderResourceMask >>= 1)
				if (resetShaderResourceMask & 0x1)
					ApplyShaderResource(i);
	}
};

typedef ShaderStageStateManager<ShaderStageStateTraits::VertexShader> VertexShaderStateManager;
typedef ShaderStageStateManager<ShaderStageStateTraits::HullShader> HullShaderStateManager;
typedef ShaderStageStateManager<ShaderStageStateTraits::DomainShader> DomainShaderStateManager;
typedef ShaderStageStateManager<ShaderStageStateTraits::GeometryShader> GeometryShaderStateManager;
typedef ShaderStageStateManager<ShaderStageStateTraits::PixelShader> PixelShaderStateManager;
typedef ShaderStageStateManager<ShaderStageStateTraits::ComputeShader> ComputeShaderStateManager;

/// State manager interface.
class StateManager : public beGraphics::StateManager
{
private:
	lean::com_ptr<ID3D11DeviceContext> m_pContext;

	StateSetup m_setup;

	VertexShaderStateManager m_VSManager;
	HullShaderStateManager m_HSManager;
	DomainShaderStateManager m_DSManager;
	GeometryShaderStateManager m_GSManager;
	PixelShaderStateManager m_PSManager;
	ComputeShaderStateManager m_CSManager;

	uint4 m_invalidMask;
	uint4 m_overrideMask;

	/// Applies the stored rasterizer state.
	LEAN_INLINE void ApplyRasterizerState() const
	{
		m_pContext->RSSetState(m_setup.RasterizerState);
	}
	/// Applies the stored depth-stencil state.
	LEAN_INLINE void ApplyDepthStencilState() const
	{
		m_pContext->OMSetDepthStencilState(m_setup.DepthStencilState, m_setup.StencilRef);
	}
	/// Applies the stored blend state.
	LEAN_INLINE void ApplyBlendState() const
	{
		m_pContext->OMSetBlendState(m_setup.BlendState, m_setup.BlendFactor, m_setup.SampleMask);
	}
	/// Applies the stored render target state.
	LEAN_INLINE void ApplyRenderTargets() const
	{
		m_pContext->OMSetRenderTargetsAndUnorderedAccessViews(m_setup.RenderTargetCount, &m_setup.RenderTargets[0].get(),
				m_setup.DepthStencilTarget,
				m_setup.RenderTargetCount, D3D11_KEEP_UNORDERED_ACCESS_VIEWS, nullptr, nullptr
			);
	}

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API StateManager(ID3D11DeviceContext *pContext);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~StateManager();

	/// Sets the given states.
	BE_GRAPHICS_DX11_API void Set(const beGraphics::StateSetup& setup);
	/// Gets all stored states.
	LEAN_INLINE const StateSetup& Get() const { return m_setup; }

	/// Records overridden state.
	BE_GRAPHICS_DX11_API void RecordOverridden();
	/// Clears the given states.
	BE_GRAPHICS_DX11_API void Clear(uint4 stateMask = StateMasks::AllStates);
	/// Clears the given states.
	BE_GRAPHICS_DX11_API void ClearBindings() LEAN_OVERRIDE { Clear(); };

	/// Invalidates the given states.
	LEAN_INLINE void Invalidate(uint4 stateMask = StateMasks::AllStates) { m_invalidMask |= stateMask; }
	/// Overrides the given states.
	LEAN_INLINE void Override(uint4 stateMask = StateMasks::AllStates) { m_overrideMask |= stateMask; Invalidate(stateMask); }
	/// Gets overridden states.
	LEAN_INLINE uint4 Overridden() const {return m_overrideMask; }
	/// Revers the given overridden states.
	void Revert(uint4 stateMask = StateMasks::AllStates)
	{
		m_overrideMask &= ~stateMask;

		if (stateMask & StateMasks::ShaderStates)
		{
			if (stateMask & StateMasks::VertexShader)
				m_VSManager.Revert();
			if (stateMask & StateMasks::HullShader)
				m_HSManager.Revert();
			if (stateMask & StateMasks::DomainShader)
				m_DSManager.Revert();
			if (stateMask & StateMasks::GeometryShader)
				m_GSManager.Revert();
			if (stateMask & StateMasks::PixelShader)
				m_PSManager.Revert();
			if (stateMask & StateMasks::ComputeShader)
				m_CSManager.Revert();
		}
	}

	/// Gets the vertex shader state manager.
	LEAN_INLINE VertexShaderStateManager& VS() { return m_VSManager; }
	/// Gets the hull shader state manager.
	LEAN_INLINE HullShaderStateManager& HS() { return m_HSManager; }
	/// Gets the domain shader state manager.
	LEAN_INLINE DomainShaderStateManager& DS() { return m_DSManager; }
	/// Gets the geometry shader state manager.
	LEAN_INLINE GeometryShaderStateManager& GS() { return m_GSManager; }
	/// Gets the pixel shader state manager.
	LEAN_INLINE PixelShaderStateManager& PS() { return m_PSManager; }
	/// Gets the compute shader state manager.
	LEAN_INLINE ComputeShaderStateManager& CS() { return m_CSManager; }

	/// Resets invalid states in the given device context.
	void Reset()
	{
		uint4 resetMask = m_invalidMask & ~m_overrideMask;
		m_invalidMask &= ~resetMask;

		// Do not reset anything that has not been recorded
		resetMask &= m_setup.StateMask;

		if (resetMask & StateMasks::AllStates)
		{
			if (resetMask & StateMasks::ShaderStates)
			{
				if (resetMask & StateMasks::VertexShader)
					m_VSManager.ResetShaderState();
				if (resetMask & StateMasks::HullShader)
					m_HSManager.ResetShaderState();
				if (resetMask & StateMasks::DomainShader)
					m_DSManager.ResetShaderState();
				if (resetMask & StateMasks::GeometryShader)
					m_GSManager.ResetShaderState();
				if (resetMask & StateMasks::PixelShader)
					m_PSManager.ResetShaderState();
				if (resetMask & StateMasks::ComputeShader)
					m_CSManager.ResetShaderState();
			}

			if (resetMask & StateMasks::PipelineStates)
			{
				if (resetMask & StateMasks::RasterizerState)
					ApplyRasterizerState();
				if (resetMask & StateMasks::DepthStencilState)
					ApplyDepthStencilState();
				if (resetMask & StateMasks::BlendState)
					ApplyBlendState();
				if (resetMask & StateMasks::RenderTargets)
					ApplyRenderTargets();
			}
		}
	}

	/// Gets the implementation identifier.
	ImplementationID GetImplementationID() const { return DX11Implementation; };

	/// Gets the device context.
	LEAN_INLINE ID3D11DeviceContext* GetContext() const { return m_pContext; };
};

template <> struct ToImplementationDX11<beGraphics::StateManager> { typedef StateManager Type; };

} // namespace

} // namespace

#endif