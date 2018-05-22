#include <stdafx.h>
#include <DEMO.h>
#include <DX11_BLEND_STATE.h>

void DX11_BLEND_STATE::Release()
{
	SAFE_RELEASE(blendState);
}

bool DX11_BLEND_STATE::Create(const BLEND_DESC &desc)
{
	this->desc = desc;
	D3D11_BLEND_DESC blendStateDesc;
	ZeroMemory(&blendStateDesc,sizeof(D3D11_BLEND_DESC));
	blendStateDesc.AlphaToCoverageEnable = FALSE;
	blendStateDesc.IndependentBlendEnable = FALSE;
	blendStateDesc.RenderTarget[0].BlendEnable = desc.blend;
	blendStateDesc.RenderTarget[0].SrcBlend = (D3D11_BLEND)desc.srcColorBlend;
	blendStateDesc.RenderTarget[0].DestBlend = (D3D11_BLEND)desc.dstColorBlend;
	blendStateDesc.RenderTarget[0].BlendOp = (D3D11_BLEND_OP)desc.blendColorOp;
	blendStateDesc.RenderTarget[0].SrcBlendAlpha = (D3D11_BLEND)desc.srcAlphaBlend;
	blendStateDesc.RenderTarget[0].DestBlendAlpha = (D3D11_BLEND)desc.dstAlphaBlend;
	blendStateDesc.RenderTarget[0].BlendOpAlpha = (D3D11_BLEND_OP)desc.blendAlphaOp;
	blendStateDesc.RenderTarget[0].RenderTargetWriteMask = desc.colorMask;
	if(DEMO::renderer->GetDevice()->CreateBlendState(&blendStateDesc,&blendState)!=S_OK)
		return false;

	return true;
}

void DX11_BLEND_STATE::Set() const
{
	DEMO::renderer->GetDeviceContext()->OMSetBlendState(blendState,desc.constBlendColor,0xFFFFFFFF);
}


