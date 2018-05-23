/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_DEVICE_CONTEXT_DX11
#define BE_GRAPHICS_DEVICE_CONTEXT_DX11

#include "beGraphics.h"
#include "../beDeviceContext.h"
#include <beCore/beWrapper.h>
#include <lean/smart/com_ptr.h>
#include <D3D11.h>

namespace beGraphics
{

namespace DX11
{

struct ShaderStages
{
	enum T
	{
		VS = 0x1,
		PS = 0x2,
		GS = 0x4,
		HS = 0x8,
		DS = 0x10,
		CS = 0x20,

		VSPS = VS | PS,
		VSPSGS = VSPS | GS,
		Graphics = VSPSGS | HS | DS,
		All = Graphics | CS,

		MaxValue = 0x7fffffff
	};
};

/// Sets the given constant buffers.
template <int Stages>
inline void SetConstantBuffers(ID3D11DeviceContext *context, UINT startSlot, UINT numBuffers, ID3D11Buffer *const *constantBuffers)
{
	if (Stages & ShaderStages::VS) context->VSSetConstantBuffers(startSlot, numBuffers, constantBuffers);
	if (Stages & ShaderStages::PS) context->PSSetConstantBuffers(startSlot, numBuffers, constantBuffers);
	if (Stages & ShaderStages::GS) context->GSSetConstantBuffers(startSlot, numBuffers, constantBuffers);
	if (Stages & ShaderStages::HS) context->HSSetConstantBuffers(startSlot, numBuffers, constantBuffers);
	if (Stages & ShaderStages::DS) context->DSSetConstantBuffers(startSlot, numBuffers, constantBuffers);
	if (Stages & ShaderStages::CS) context->CSSetConstantBuffers(startSlot, numBuffers, constantBuffers);
}
/// Sets the given constant buffers.
inline void SetConstantBuffers(ID3D11DeviceContext *context, UINT startSlot, UINT numBuffers, ID3D11Buffer *const *constantBuffers)
{
	SetConstantBuffers<ShaderStages::All>(context, startSlot, numBuffers, constantBuffers);
}

/// Sets the given resource views.
template <int Stages>
inline void SetShaderResources(ID3D11DeviceContext *context, UINT startSlot, UINT numBuffers, ID3D11ShaderResourceView *const *resources)
{
	if (Stages & ShaderStages::VS) context->VSSetShaderResources(startSlot, numBuffers, resources);
	if (Stages & ShaderStages::PS) context->PSSetShaderResources(startSlot, numBuffers, resources);
	if (Stages & ShaderStages::GS) context->GSSetShaderResources(startSlot, numBuffers, resources);
	if (Stages & ShaderStages::HS) context->HSSetShaderResources(startSlot, numBuffers, resources);
	if (Stages & ShaderStages::DS) context->DSSetShaderResources(startSlot, numBuffers, resources);
	if (Stages & ShaderStages::CS) context->CSSetShaderResources(startSlot, numBuffers, resources);
}
/// Sets the given resource views.
inline void SetShaderResources(ID3D11DeviceContext *context, UINT startSlot, UINT numBuffers, ID3D11ShaderResourceView *const *resources)
{
	SetShaderResources<ShaderStages::All>(context, startSlot, numBuffers, resources);
}

/// Clears all pixel shader output resources.
BE_GRAPHICS_DX11_API void UnbindAllRenderTargets(ID3D11DeviceContext *context);
/// Clears all compute shader output resources.
BE_GRAPHICS_DX11_API void UnbindAllComputeTargets(ID3D11DeviceContext *context);
/// Clears all output resources.
BE_GRAPHICS_DX11_API void UnbindAllTargets(ID3D11DeviceContext *context);

/// Clears all shader resources.
BE_GRAPHICS_DX11_API void UnbindAllShaderResources(ID3D11DeviceContext *context);

/// Unbinds everything.
BE_GRAPHICS_DX11_API void UnbindAll(ID3D11DeviceContext *context);

/// Device context implementation.
class DeviceContext : public beCore::IntransitiveWrapper<ID3D11DeviceContext, DeviceContext>,
	public beGraphics::DeviceContext
{
private:
	lean::com_ptr<ID3D11DeviceContext> m_pContext;
	
public:
	/// Constructor.
	LEAN_INLINE DeviceContext(ID3D11DeviceContext *pContext)
		: m_pContext( LEAN_ASSERT_NOT_NULL(pContext) ) { }
	
	/// Gets the D3D device context.
	LEAN_INLINE ID3D11DeviceContext*const& GetInterface() const { return m_pContext.get(); }
	/// Gets the D3D device context.
	LEAN_INLINE ID3D11DeviceContext*const& GetContext() const { return m_pContext.get(); }

	/// Clears all state.
	BE_GRAPHICS_DX11_API void ClearState() LEAN_OVERRIDE { m_pContext->ClearState(); };

	/// Gets the implementation identifier.
	LEAN_INLINE ImplementationID GetImplementationID() const { return DX11Implementation; };
};

template <> struct ToImplementationDX11<beGraphics::DeviceContext> { typedef DeviceContext Type; };

} // namespace

} // namespace

#endif