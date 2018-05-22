#include <stdafx.h>
#include <DEMO.h>
#include <DX11_TEXTURE.h>

void DX11_TEXTURE::Release()
{
	SAFE_RELEASE(texture);
	SAFE_RELEASE(shaderResourceView);
	SAFE_RELEASE(unorderedAccessView);
}

bool DX11_TEXTURE::LoadFromFile(const char *fileName,DX11_SAMPLER *sampler)
{	
	strcpy(name,fileName);
	char filePath[DEMO_MAX_FILEPATH];
	if(!DEMO::fileManager->GetFilePath(fileName,filePath))
		return false;

	if(D3DX11CreateTextureFromFile(DEMO::renderer->GetDevice(),filePath,NULL,NULL,&texture,NULL)!=S_OK)
		return false;

	if(DEMO::renderer->GetDevice()->CreateShaderResourceView(texture,NULL,&shaderResourceView)!=S_OK)
		return false;

	if(!sampler)
		this->sampler = DEMO::renderer->GetSampler(TRILINEAR_SAMPLER_ID);
	else
		this->sampler = sampler;

	return true;
}

bool DX11_TEXTURE::CreateRenderable(int width,int height,int depth,texFormats format,DX11_SAMPLER *sampler,bool useUAV)
{
	strcpy(name,"renderTargetTexture");
	
	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc,sizeof(textureDesc));
	textureDesc.Width = width;
	textureDesc.Height = height; 
	textureDesc.ArraySize = depth;
	textureDesc.MipLevels = 1;
	textureDesc.Format = (DXGI_FORMAT)format;
	textureDesc.SampleDesc.Count = 1;
	textureDesc.SampleDesc.Quality = 0;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;
	if(format!=TEX_FORMAT_DEPTH24)
		textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	else 
		textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL;
	if(useUAV)
		textureDesc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.MiscFlags = 0;

	if(DEMO::renderer->GetDevice()->CreateTexture2D(&textureDesc,NULL,(ID3D11Texture2D**)&texture)!=S_OK)
		return false;

	if(depth==1)
	{
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc; 
		if(format!=TEX_FORMAT_DEPTH24)
			srvDesc.Format = textureDesc.Format; 
		else
			srvDesc.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; 
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MipLevels = 1;
		srvDesc.Texture2D.MostDetailedMip = 0;
		if(DEMO::renderer->GetDevice()->CreateShaderResourceView(texture,&srvDesc,&shaderResourceView)!=S_OK)
			return false;
	}
	else
	{
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc; 
		if(format!=TEX_FORMAT_DEPTH24)
			srvDesc.Format = textureDesc.Format; 
		else
			srvDesc.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; 
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
		srvDesc.Texture2DArray.MipLevels = 1;
		srvDesc.Texture2DArray.MostDetailedMip = 0;
		srvDesc.Texture2DArray.ArraySize = depth;
		srvDesc.Texture2DArray.FirstArraySlice = 0;
		if(DEMO::renderer->GetDevice()->CreateShaderResourceView(texture,&srvDesc,&shaderResourceView)!=S_OK)
			return false;
	}

	if(useUAV)
	{
		if(DEMO::renderer->GetDevice()->CreateUnorderedAccessView(texture,NULL,&unorderedAccessView)!=S_OK)
				return false;
	}
	
	if(!sampler)
		this->sampler = DEMO::renderer->GetSampler(LINEAR_SAMPLER_ID);
	else
		this->sampler = sampler;

	return true;
}

void DX11_TEXTURE::Bind(textureBP bindingPoint,shaderTypes shaderType) const
{
	switch(shaderType)
	{
	case VERTEX_SHADER:
		DEMO::renderer->GetDeviceContext()->VSSetShaderResources(bindingPoint,1,&shaderResourceView);
		break;

	case GEOMETRY_SHADER:
		DEMO::renderer->GetDeviceContext()->GSSetShaderResources(bindingPoint,1,&shaderResourceView);
		break;

	case FRAGMENT_SHADER: 
		DEMO::renderer->GetDeviceContext()->PSSetShaderResources(bindingPoint,1,&shaderResourceView);
		break;

	case COMPUTE_SHADER: 
		DEMO::renderer->GetDeviceContext()->CSSetShaderResources(bindingPoint,1,&shaderResourceView);
		break;
	}

	sampler->Bind(shaderType,bindingPoint);
}





