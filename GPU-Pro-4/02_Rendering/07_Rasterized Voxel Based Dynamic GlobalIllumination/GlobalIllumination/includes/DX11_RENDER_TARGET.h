#ifndef DX11_RENDER_TARGET_H
#define DX11_RENDER_TARGET_H

#include <render_states.h>

// max number of color-buffers, which can be attached to 1 render-target
#define MAX_NUM_COLOR_BUFFERS 8

class DX11_TEXTURE;
class DX11_SAMPLER;
class RENDER_TARGET_CONFIG;

// DX11_RENDER_TARGET
//   Render-target to render/ write into. Can be configured via RENDER_TARGET_CONFIG for each draw-call/ dispatch.
class DX11_RENDER_TARGET
{
public:
	DX11_RENDER_TARGET()
	{
		width = 0;
		height = 0;
		depth = 0;
		depthStencil = false;
		clearMask = 0;
		numColorBuffers = 0;
		clearTarget = true;
		renderTargetViews = NULL;
		frameBufferTextures = NULL;	
		depthStencilView = NULL;
		depthStencilTexture = NULL;	
	}

	~DX11_RENDER_TARGET()
	{
		Release();
	}

	void Release();

	bool Create(int width,int height,int depth,texFormats format=TEX_FORMAT_RGB16F,bool depthStencil=false,
		          int numColorBuffers=1,DX11_SAMPLER *sampler=NULL,bool useUAV=false);

	bool CreateBackBuffer();

	void Bind(RENDER_TARGET_CONFIG *rtConfig=NULL);

	// indicate, that render-target should be cleared
	void Reset()
	{
		clearTarget = true;
	}

	void Clear(unsigned int newClearMask) const;

	DX11_TEXTURE* GetTexture(int index=0) const;

	DX11_TEXTURE* GetDepthStencilTexture() const;

	int GetWidth() const
	{
		return width;
	}

	int GetHeight() const
	{
		return height;
	}

	int GetDepth() const
	{
		return depth;
	}

private:	
	int width,height,depth;
	bool depthStencil;
	int numColorBuffers;
	unsigned int clearMask;
	bool clearTarget;	
	ID3D11RenderTargetView **renderTargetViews;
	DX11_TEXTURE *frameBufferTextures;
	ID3D11DepthStencilView *depthStencilView;
	DX11_TEXTURE *depthStencilTexture;
	D3D11_VIEWPORT viewport;

};

#endif