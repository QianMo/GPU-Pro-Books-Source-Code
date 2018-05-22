#include <stdafx.h>
#include <DEMO.h>
#include <DX11_SAMPLER.h>

void DX11_SAMPLER::Release()
{
	SAFE_RELEASE(sampler);
}

bool DX11_SAMPLER::Create(const SAMPLER_DESC &desc)
{
	this->desc = desc;
	D3D11_SAMPLER_DESC samplerDesc;
	ZeroMemory(&samplerDesc,sizeof(samplerDesc));
	samplerDesc.Filter = (D3D11_FILTER)desc.filter;
	samplerDesc.AddressU = (D3D11_TEXTURE_ADDRESS_MODE)desc.adressU;
	samplerDesc.AddressV = (D3D11_TEXTURE_ADDRESS_MODE)desc.adressV;
	samplerDesc.AddressW = (D3D11_TEXTURE_ADDRESS_MODE)desc.adressW;
	samplerDesc.MipLODBias = desc.lodBias;
	samplerDesc.MaxAnisotropy = desc.maxAnisotropy;
	samplerDesc.ComparisonFunc = (D3D11_COMPARISON_FUNC)desc.compareFunc;
	memcpy(samplerDesc.BorderColor,desc.borderColor,sizeof(COLOR));
	samplerDesc.MinLOD = (desc.minLOD+FLT_MAX)*0.5f;
	samplerDesc.MaxLOD = (desc.maxLOD+FLT_MAX)*0.5f;
	if(DEMO::renderer->GetDevice()->CreateSamplerState(&samplerDesc,&sampler)!=S_OK)
		return false;
	
	return true;
}

void DX11_SAMPLER::Bind(shaderTypes shaderType,textureBP bindingPoint) const
{
	switch(shaderType)
	{
	case VERTEX_SHADER:
		DEMO::renderer->GetDeviceContext()->VSSetSamplers(bindingPoint,1,&sampler);
		break;

	case GEOMETRY_SHADER:
		DEMO::renderer->GetDeviceContext()->GSSetSamplers(bindingPoint,1,&sampler);
		break;

	case FRAGMENT_SHADER:
    DEMO::renderer->GetDeviceContext()->PSSetSamplers(bindingPoint,1,&sampler);
		break;

	case COMPUTE_SHADER:
		DEMO::renderer->GetDeviceContext()->CSSetSamplers(bindingPoint,1,&sampler);
		break;
	}	
}


