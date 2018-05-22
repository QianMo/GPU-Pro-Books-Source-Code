#include <stdafx.h>
#include <DEMO.h>
#include <DX11_STRUCTURED_BUFFER.h>
#include <DX11_RENDER_TARGET.h>

void DX11_RENDER_TARGET::Release()
{
	SAFE_DELETE_ARRAY(frameBufferTextures);
	if(renderTargetViews)
	{
		for(int i=0;i<numColorBuffers;i++)
			SAFE_RELEASE(renderTargetViews[i]);
		SAFE_DELETE_ARRAY(renderTargetViews);
	}
  SAFE_DELETE(depthStencilTexture);
	SAFE_RELEASE(depthStencilView);
}

bool DX11_RENDER_TARGET::Create(int width,int height,int depth,texFormats format,bool depthStencil,int numColorBuffers,
																DX11_SAMPLER *sampler,bool useUAV)
{
	this->width = width;
	this->height = height;
	this->depth = depth;	
	this->depthStencil = depthStencil;
	if((numColorBuffers<0)||(numColorBuffers>MAX_NUM_COLOR_BUFFERS))
		return false;
	this->numColorBuffers = numColorBuffers;
	if((numColorBuffers>0)||(depthStencil))
	{
		viewport.TopLeftX = 0.0f;
		viewport.TopLeftY = 0.0f;
		viewport.Width = (float)width;
		viewport.Height = (float)height;
		viewport.MinDepth = 0.0f;
		viewport.MaxDepth = 1.0f;
	
		if(numColorBuffers>0)
		{
			clearMask = COLOR_CLEAR_BIT;
			frameBufferTextures = new DX11_TEXTURE[numColorBuffers];
			if(!frameBufferTextures)
				return false;
			renderTargetViews = new ID3D11RenderTargetView*[numColorBuffers];
			if(!renderTargetViews)
				return false;
			for(int i=0;i<numColorBuffers;i++)
			{
				if(!frameBufferTextures[i].CreateRenderable(width,height,depth,format,sampler,useUAV))
					return false;
				if(DEMO::renderer->GetDevice()->CreateRenderTargetView(frameBufferTextures[i].texture,NULL,&renderTargetViews[i])!=S_OK)
					return false;
			}
		}
		
		if(depthStencil)
		{
			clearMask |= DEPTH_CLEAR_BIT | STENCIL_CLEAR_BIT;
			depthStencilTexture = new DX11_TEXTURE;
			if(!depthStencilTexture)
				return false;
			if(!depthStencilTexture->CreateRenderable(width,height,depth,TEX_FORMAT_DEPTH24,sampler))
				return false;
			D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
			ZeroMemory(&descDSV,sizeof(descDSV));
			descDSV.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
			descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
			descDSV.Texture2D.MipSlice = 0;
			if(DEMO::renderer->GetDevice()->CreateDepthStencilView(depthStencilTexture->texture,&descDSV,&depthStencilView)!=S_OK)
				return false;
		}
	}

	return true;
}

bool DX11_RENDER_TARGET::CreateBackBuffer()
{
	width = SCREEN_WIDTH;
	height = SCREEN_HEIGHT;  
	depth = 1;
	depthStencil = true;
	clearMask = COLOR_CLEAR_BIT | DEPTH_CLEAR_BIT | STENCIL_CLEAR_BIT;
	numColorBuffers = 1;	

	viewport.TopLeftX = 0.0f;
	viewport.TopLeftY = 0.0f;
	viewport.Width = (float)width;
	viewport.Height = (float)height;
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	ID3D11Texture2D* backBufferTexture = NULL;
	if(DEMO::renderer->GetSwapChain()->GetBuffer(0,__uuidof( ID3D11Texture2D),(LPVOID*)&backBufferTexture)!=S_OK)
		return false;
	renderTargetViews = new ID3D11RenderTargetView*[numColorBuffers];
	if(!renderTargetViews)
		return false;
	if(DEMO::renderer->GetDevice()->CreateRenderTargetView(backBufferTexture,NULL,&renderTargetViews[0])!=S_OK)
	{
		backBufferTexture->Release();
		return false;
	}
	backBufferTexture->Release();

	depthStencilTexture = new DX11_TEXTURE;
	if(!depthStencilTexture)
		return false;
	if(!depthStencilTexture->CreateRenderable(width,height,depth,TEX_FORMAT_DEPTH24))
		return false;
	D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
	ZeroMemory(&descDSV,sizeof(descDSV));
	descDSV.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	descDSV.Texture2D.MipSlice = 0;
	if(DEMO::renderer->GetDevice()->CreateDepthStencilView(depthStencilTexture->texture,&descDSV,&depthStencilView)!=S_OK)
		return false;

	return true;
}

void DX11_RENDER_TARGET::Bind(RENDER_TARGET_CONFIG *rtConfig)
{
	if((numColorBuffers>0)||(depthStencil))
	  DEMO::renderer->GetDeviceContext()->RSSetViewports(1,&viewport);

	if(!rtConfig)
	{
		if((numColorBuffers>0)||(depthStencil))
		  DEMO::renderer->GetDeviceContext()->OMSetRenderTargets(numColorBuffers,renderTargetViews,depthStencilView);
	}
	else
	{
    RT_CONFIG_DESC rtConfigDesc = rtConfig->GetDesc();
		if(!rtConfigDesc.computeTarget)
		{
			if(rtConfigDesc.numStructuredBuffers==0)
			{
				DEMO::renderer->GetDeviceContext()->OMSetRenderTargets(rtConfigDesc.numColorBuffers, 
					&renderTargetViews[rtConfigDesc.firstColorBufferIndex],depthStencilView);
			}
			else
			{
				assert(rtConfigDesc.numStructuredBuffers<=MAX_NUM_SB_BUFFERS);
				ID3D11UnorderedAccessView *sbUnorderedAccessViews[MAX_NUM_SB_BUFFERS];
				for(int i=0;i<rtConfigDesc.numStructuredBuffers;i++)
					sbUnorderedAccessViews[i] = ((DX11_STRUCTURED_BUFFER*)rtConfigDesc.structuredBuffers[i])->GetUnorderdAccessView();
				DEMO::renderer->GetDeviceContext()->OMSetRenderTargetsAndUnorderedAccessViews(numColorBuffers, 
					&renderTargetViews[rtConfigDesc.firstColorBufferIndex],depthStencilView,rtConfigDesc.numColorBuffers,
					rtConfigDesc.numStructuredBuffers,sbUnorderedAccessViews,NULL);
			}		
		}
		else
		{
      DEMO::renderer->GetDeviceContext()->OMSetRenderTargets(0,NULL,NULL);	
			if(rtConfigDesc.numStructuredBuffers==0)
			{
				assert(rtConfigDesc.numColorBuffers<=MAX_NUM_COLOR_BUFFERS);
				ID3D11UnorderedAccessView *sbUnorderedAccessViews[MAX_NUM_COLOR_BUFFERS];
				for(int i=0;i<rtConfigDesc.numColorBuffers;i++)
					sbUnorderedAccessViews[i] = frameBufferTextures[i].GetUnorderdAccessView();
				DEMO::renderer->GetDeviceContext()->CSSetUnorderedAccessViews(0,rtConfigDesc.numColorBuffers,&sbUnorderedAccessViews[rtConfigDesc.firstColorBufferIndex],NULL);
			}
			else
			{
				assert(rtConfigDesc.numStructuredBuffers<=MAX_NUM_SB_BUFFERS);
				ID3D11UnorderedAccessView *sbUnorderedAccessViews[MAX_NUM_SB_BUFFERS];
				for(int i=0;i<rtConfigDesc.numStructuredBuffers;i++)
				  sbUnorderedAccessViews[i] = ((DX11_STRUCTURED_BUFFER*)rtConfigDesc.structuredBuffers[i])->GetUnorderdAccessView();
				DEMO::renderer->GetDeviceContext()->CSSetUnorderedAccessViews(0,rtConfigDesc.numStructuredBuffers,sbUnorderedAccessViews,NULL);
			}		
		}
	}

	if(clearTarget)
	{
		Clear(clearMask);
		clearTarget = false;
	}
}

void DX11_RENDER_TARGET::Clear(unsigned int newClearMask) const
{
	if((numColorBuffers>0)||(depthStencil))
	{
		if(newClearMask & COLOR_CLEAR_BIT)
		{
			for(int i=0;i<numColorBuffers;i++)
				DEMO::renderer->GetDeviceContext()->ClearRenderTargetView(renderTargetViews[i],CLEAR_COLOR);
	    newClearMask &= ~COLOR_CLEAR_BIT;
		}
		if(newClearMask!=0)
			DEMO::renderer->GetDeviceContext()->ClearDepthStencilView(depthStencilView,newClearMask,CLEAR_DEPTH,CLEAR_STENCIL);
	}
}

DX11_TEXTURE* DX11_RENDER_TARGET::GetTexture(int index) const
{
	if((index<0)||(index>=numColorBuffers))
		return NULL;
	return &frameBufferTextures[index];
}

DX11_TEXTURE* DX11_RENDER_TARGET::GetDepthStencilTexture() const
{
	return depthStencilTexture;
}





