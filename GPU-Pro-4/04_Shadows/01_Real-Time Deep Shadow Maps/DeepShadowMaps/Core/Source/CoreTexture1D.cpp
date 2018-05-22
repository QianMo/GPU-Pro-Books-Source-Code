#include "Core.h"

CoreTexture1D::CoreTexture1D()
{
	width = 0;
	texture = NULL;
}

// Create a texture from memory
CoreResult CoreTexture1D::init(Core* core, BYTE** data, UINT width, UINT textureCount, UINT mipLevels,
							   DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags)
{
	BYTE** dataArray = NULL;

	this->core = core;
	this->mipLevels = mipLevels;
	this->cpuAccessFlags = cpuAccessFlags;
	this->usage = usage;
	this->bindFlags = bindFlags;
	this->textureCount = textureCount;
	this->format = format;
	this->width = width;
	this->miscFlags = miscFlags;
	
	if(data)
	{
		dataArray = new BYTE* [textureCount];
		ZeroMemory(dataArray, sizeof(BYTE*) * textureCount);
		for(UINT ui = 0 ; ui < textureCount ; ui++)
		{
			if(!data[ui])
			{
				CoreLog::Information(L"Data[%d] == NULL , skipping!", ui);
				continue;
			}
			

			dataArray[ui] = data[ui];
		}
	}

	CoreResult result = createAndFillTexture(dataArray);
	
	if(data)
		delete dataArray;

	return result;
}


// As the name says...
CoreResult CoreTexture1D::createAndFillTexture(BYTE** data)
{
	if(core)
	{
		D3D11_TEXTURE1D_DESC texDesc;
		ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE1D_DESC));

		texDesc.Width = width;
		texDesc.MipLevels = mipLevels;
		texDesc.Format = format;
		texDesc.ArraySize = textureCount;
		texDesc.Usage = usage;
		texDesc.BindFlags = bindFlags;
		texDesc.CPUAccessFlags = cpuAccessFlags;
		texDesc.MiscFlags = miscFlags;

		D3D11_SUBRESOURCE_DATA *subResData = NULL;
		if(data)
		{
			subResData = new D3D11_SUBRESOURCE_DATA[textureCount];

			for(UINT ui = 0 ; ui < textureCount ; ui++)
			{
				subResData[ui].pSysMem = data[ui];
				subResData[ui].SysMemSlicePitch = 0;
				subResData[ui].SysMemPitch = GetNumberOfBytesFromDXGIFormt(format) * width;
			}
		}

		
		HRESULT result = core->GetDevice()->CreateTexture1D(&texDesc, subResData, &texture);
		
		delete subResData;
		// create the texture
		if(FAILED(result))
		{
			CoreLog::Information(L"Couldn't create D3D Texture, HRESULT = %x!", result);
			return CORE_MISC_ERROR;
		}
		return result;
	}
	else
		texture = NULL;
	

	return CORE_OK;
}

// CleanUp
void CoreTexture1D::finalRelease()
{
	SAFE_RELEASE(texture);
}

// Retrieves the RenderTargetView from the texture
CoreResult CoreTexture1D::CreateRenderTargetView(D3D11_RENDER_TARGET_VIEW_DESC* rtvDesc, ID3D11RenderTargetView** rtv)
{
	HRESULT result = core->GetDevice()->CreateRenderTargetView(texture, rtvDesc, rtv);
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create RenderTargetView, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}

// Retrieves the DepthStencilView from the texture
CoreResult CoreTexture1D::CreateDepthStencilView(D3D11_DEPTH_STENCIL_VIEW_DESC* dsvDesc, ID3D11DepthStencilView** dsv)
{
	HRESULT result = core->GetDevice()->CreateDepthStencilView(texture, dsvDesc, dsv);
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create DepthStencilView, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}


// Creates a ShaderResourceView with this texture as resource
CoreResult CoreTexture1D::CreateShaderResourceView(D3D11_SHADER_RESOURCE_VIEW_DESC* srvDesc, ID3D11ShaderResourceView** srv)
{
	HRESULT result = core->GetDevice()->CreateShaderResourceView(texture, srvDesc, srv);
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create DepthStencilView, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}
