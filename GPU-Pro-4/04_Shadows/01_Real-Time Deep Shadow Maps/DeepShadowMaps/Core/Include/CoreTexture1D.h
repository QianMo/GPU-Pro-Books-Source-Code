#pragma once

#include <istream>

#include "CoreError.h"
#include "CoreColor.h"
#include "ICoreBase.h"
#include "CoreLog.h"
#include <d3d11.h>

class Core;

class CoreTexture1D : public ICoreBase
{
friend Core;
protected:
	// Init constructor
	CoreTexture1D();

	ID3D11Texture1D* texture;
	Core* core;

	UINT width;
	UINT mipLevels;
	UINT cpuAccessFlags;
	UINT miscFlags;
	UINT textureCount;
	D3D11_USAGE usage;
	UINT bindFlags;
	DXGI_FORMAT format;

	// Create a texture from memory
	CoreResult init(Core* core, BYTE** data, UINT width, UINT textureCount, UINT mipLevels,
				    DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags);

	CoreResult CoreTexture1D::createAndFillTexture(BYTE** data);
	
	// CleanUp
	virtual void finalRelease();
public:
	// Create a RenderTargetView
		CoreResult CreateRenderTargetView(D3D11_RENDER_TARGET_VIEW_DESC* rtvDesc, ID3D11RenderTargetView** rtv);
		// Create a DepthStencilView 
		CoreResult CreateDepthStencilView(D3D11_DEPTH_STENCIL_VIEW_DESC* dsvDesc, ID3D11DepthStencilView** dsv);
		// Create a ShaderResourceView
		CoreResult CreateShaderResourceView(D3D11_SHADER_RESOURCE_VIEW_DESC* srvDesc, ID3D11ShaderResourceView** srv);


		inline DXGI_FORMAT GetFormat()						{ return format; }

};