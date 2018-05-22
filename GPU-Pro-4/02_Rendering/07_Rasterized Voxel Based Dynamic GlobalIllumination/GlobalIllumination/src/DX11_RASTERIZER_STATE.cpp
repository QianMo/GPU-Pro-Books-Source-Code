#include <stdafx.h>
#include <DEMO.h>
#include <DX11_RASTERIZER_STATE.h>

void DX11_RASTERIZER_STATE::Release()
{
	SAFE_RELEASE(rasterizerState);
}

bool DX11_RASTERIZER_STATE::Create(const RASTERIZER_DESC &desc)
{
	this->desc = desc;
	D3D11_RASTERIZER_DESC rasterDesc;
	ZeroMemory(&rasterDesc,sizeof(D3D11_RASTERIZER_DESC));
	rasterDesc.FillMode = (D3D11_FILL_MODE)desc.fillMode;
	rasterDesc.CullMode = (D3D11_CULL_MODE)desc.cullMode;
	rasterDesc.FrontCounterClockwise = TRUE;
	rasterDesc.ScissorEnable = desc.scissorTest;
  rasterDesc.MultisampleEnable = desc.multisampleEnable;
	if(DEMO::renderer->GetDevice()->CreateRasterizerState(&rasterDesc,&rasterizerState)!=S_OK)
		return false;

	return true;
}

void DX11_RASTERIZER_STATE::Set() const
{
	DEMO::renderer->GetDeviceContext()->RSSetState(rasterizerState);
}


