/*******************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*******************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beDeviceContext.h"

namespace beGraphics
{

namespace DX11
{

// Unbinds all pixel shader output resources.
void UnbindAllRenderTargets(ID3D11DeviceContext *context)
{
	context->OMSetRenderTargetsAndUnorderedAccessViews(0, nullptr, nullptr, 0, 0, nullptr, nullptr);
}

// Unbinds all compute shader output resources.
void UnbindAllComputeTargets(ID3D11DeviceContext *context)
{
	static ID3D11UnorderedAccessView *const NullUAVs[D3D11_PS_CS_UAV_REGISTER_COUNT] = { nullptr };
	context->CSSetUnorderedAccessViews(0, D3D11_PS_CS_UAV_REGISTER_COUNT, NullUAVs, nullptr);
}

// Unbinds all output resources.
void UnbindAllTargets(ID3D11DeviceContext *context)
{
	UnbindAllRenderTargets(context);
	UnbindAllComputeTargets(context);
}

// Clears all shader resources.
void UnbindAllShaderResources(ID3D11DeviceContext *context)
{
	static ID3D11ShaderResourceView *const NullSRVs[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT ] = { nullptr };
	SetShaderResources(context, 0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, NullSRVs);
}

// Unbinds everything.
void UnbindAll(ID3D11DeviceContext *context)
{
	context->ClearState();
}

} // namespace

} // namespace