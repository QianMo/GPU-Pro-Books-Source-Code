#pragma once

#include "CoreError.h"
#include "CoreLog.h"
#include "CoreColor.h"
#include "CoreTexture2D.h"
#include "CoreTexture3D.h"
#include "ICoreBase.h"
#include "CoreUtils.h"
#include "CoreVector3.h"
#include "CoreMatrix4x4.h"
#include "CoreCamera.h"
#include "CoreTexture1D.h"
#include "CorePlane.h"
#include "CoreQuaternion.h"
#include <d3d11.h>

#include <vector>

class Core;

// Creates the Core
CoreResult CreateCore(HWND wndMain, UINT swapChainBufferCount, D3D_DRIVER_TYPE driver, DXGI_FORMAT colorFormat,
					  UINT refreshRateNumerator, UINT refreshRateDenominator, UINT sampleCount, 
					  UINT sampleQuality, bool windowed, IDXGIAdapter* adapter, Core** outCore);
// Create a D3D device without SwapChain
CoreResult CreateCore(D3D_DRIVER_TYPE driver, IDXGIAdapter* adapter, Core** outCore);


class Core : public ICoreBase
{
friend CoreResult CreateCore(HWND wndMain, UINT swapChainBufferCount, D3D_DRIVER_TYPE driver, DXGI_FORMAT colorFormat,
							 UINT refreshRateNumerator, UINT refreshRateDenominator, UINT sampleCount, 
							 UINT sampleQuality, bool windowed, IDXGIAdapter* adapter, Core** outCore);
friend CoreResult CreateCore(D3D_DRIVER_TYPE driver, IDXGIAdapter* adapter, Core** outCore);
public:
	// inline functions
	inline ID3D11Device*					GetDevice()					{ return device; }
	inline ID3D11DeviceContext*				GetImmediateDeviceContext()	{ return immediateDeviceContext; }
	// Only returns the default SwapChain (not created with CreateSwapChain)
	inline IDXGISwapChain*					GetSwapChain()				{ return swapChain; }

	// Sets the RenderTargets
	void SetRenderTargets(UINT numRenderTargets, ID3D11RenderTargetView** rtvs, ID3D11DepthStencilView* dsv);
	
	inline ID3D11RenderTargetView* GetOriginalRenderTargetView()		{ return renderTargetView; }
	inline ID3D11DepthStencilView* GetOriginalDepthStencilView()		{ return depthStencilView; }
	inline ID3D11RenderTargetView* GetRenderTargetView()				{ return renderTargetViewOverwrite; }
	inline ID3D11DepthStencilView* GetDepthStencilView()				{ return depthStencilViewOverwrite; }


	inline CoreTexture2D* GetBackBuffer()								{ return backBuffer;}
	inline CoreTexture2D* GetDepthStencil()								{ return depthStencil;}
	inline ID3D11ShaderResourceView* GetDepthStencilShaderResourceView(){ return depthStencilSRV; }

		// Create a texture from memory
	CoreResult CreateTexture1D(BYTE** data, UINT width, UINT textureCount, UINT mipLevels,
							   DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
							   CoreTexture1D** outTex);
	// Load a texture from a stream
	CoreResult CreateTexture2D(std::istream *in[], UINT textureCount, UINT mipLevels,
				  			   UINT cpuAccessFlag, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
							   UINT sampleCount, UINT sampleQuality, bool sRGB, CoreTexture2D** outTex);
	
	//Load multiple textures from streams
	CoreResult CreateTexture2D(const std::vector <std::istream *> &in, UINT mipLevels,
				  		   UINT cpuAccessFlag, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
						   UINT sampleCount, UINT sampleQuality, bool sRGB, CoreTexture2D** outTex);

	// Load a single texture from a stream
	CoreResult CreateTexture2D(std::istream &in, UINT mipLevels,
				  			   UINT cpuAccessFlag, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
							   UINT sampleCount, UINT sampleQuality, bool sRGB, CoreTexture2D** outTex);
	
	// Create a texture from memory
	CoreResult CreateTexture2D(BYTE** data, UINT width, UINT height, UINT textureCount, UINT mipLevels,
							   DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
							   UINT sampleCount, UINT sampleQuality, CoreTexture2D** outTex);

	// Load a 3D texture from a stream
	CoreResult CreateTexture3D(std::istream& in, UINT mipLevels,
			  	    UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags, CoreTexture3D** outTex);
	
	// Load a 3D texture from memory
	CoreResult CreateTexture3D(BYTE* data, UINT width, UINT height, UINT depth, UINT mipLevels,
			  	    DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
					CoreTexture3D** outTex);
	
	// Directly use an ID3D11Texture2D object
	CoreResult CreateTexture2D(ID3D11Texture2D* texture, CoreTexture2D** outTex);

	// Create an additional SwapChain
	CoreResult CreateSwapChain(HWND wndMain, UINT swapChainBufferCount, DXGI_FORMAT colorFormat, DXGI_FORMAT depthStencilFormat, UINT refreshRateNumerator, 
								 UINT refreshRateDenominator, UINT sampleCount, UINT sampleQuality, bool windowed,
								 IDXGISwapChain** swapChain, CoreTexture2D** backBuffer, ID3D11RenderTargetView** renderTargetView, 
								 CoreTexture2D** depthStencil, ID3D11DepthStencilView** depthStencilView);

	// Resizes an additionally created SwapChain
	CoreResult ResizeSwapChain(UINT width, UINT height, DXGI_FORMAT format, DXGI_FORMAT depthStencilFormat, IDXGISwapChain* swapChain, CoreTexture2D** backBuffer, ID3D11RenderTargetView** renderTargetView, 
								 CoreTexture2D** depthStencil, ID3D11DepthStencilView** depthStencilView);

	// Resizes the internal SwapChain
	CoreResult ResizeSwapChain(UINT width, UINT height, DXGI_FORMAT format);

	// Clear the capsuled RenderTargetView
	void ClearRenderTargetView(CoreColor &c);

	// Clear a different RenderTargetView
	void ClearRenderTargetView(ID3D11RenderTargetView* rtv, CoreColor &c);

	// Clear the capsuled DepthStencilView
	void ClearDepthStencilView(UINT clearFlags, float depth, UINT8 stencil);
	
	// Sets the internal render target/ depth stencil view
	void SetDefaultRenderTarget();

	// Overwrites the default render target. 
	// When SetDefaultRenderTarget is called, it is set to the new render target.
	void OverwriteDefaultRenderTargetView(ID3D11RenderTargetView *renderTarget);
	void RestoreDefaultRenderTargetView();

	void OverwriteDefaultDepthStencilView(ID3D11DepthStencilView *dsv);
	void RestoreDefaultDepthStencilView();

	// Saves all Viewports -> restore with RestoreViewports
	void SaveViewports();

	// Restores all Viewports
	void RestoreViewports();

protected:
	ID3D11Device*				device;
	IDXGISwapChain*				swapChain;	// Default SwapChain
	
	CoreTexture2D*				backBuffer;
	ID3D11RenderTargetView*		renderTargetView;
	ID3D11RenderTargetView*		renderTargetViewOverwrite;
	std::vector<ID3D11RenderTargetView *> overriddenRenderTargetStack;
	CoreTexture2D*				depthStencil;
	ID3D11DepthStencilView*		depthStencilView;
	ID3D11DepthStencilView*		depthStencilViewOverwrite;
	std::vector<ID3D11DepthStencilView *> overriddenDepthStencilStack;
	ID3D11ShaderResourceView*	depthStencilSRV;
	std::vector<UINT>			numSavedViewportsStack;
	std::vector<D3D11_VIEWPORT*> savedViewportsStack;
	ID3D11DeviceContext			*immediateDeviceContext;

	Core();
	// Initialize
	CoreResult init(HWND wndMain, UINT swapChainBufferCount, D3D_DRIVER_TYPE driver, DXGI_FORMAT colorFormat,
					UINT refreshRateNumerator, UINT refreshRateDenominator, UINT sampleCount, 
					UINT sampleQuality, bool windowed, IDXGIAdapter* adapter);
	// Create a D3D device without SwapChain
	CoreResult init(D3D_DRIVER_TYPE driver, IDXGIAdapter* adapter);
	
	virtual void finalRelease();
		
};