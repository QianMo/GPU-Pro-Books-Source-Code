
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _D3D11APP_H_
#define _D3D11APP_H_

#include "Direct3D11Renderer.h"
#include "../BaseApp.h"

// InitAPI flags
#define NO_SETTING_CHANGE 0x1
#define SAMPLE_BACKBUFFER 0x2

class D3D11App : public BaseApp
{
public:
	D3D11App();

	virtual bool initCaps();
	virtual bool initAPI();
	virtual void exitAPI();

	void beginFrame();
	void endFrame();

	virtual void onSize(const int w, const int h);

	bool captureScreenshot(Image &img);

protected:
	enum API_Revision
	{
		D3D11,
		D3D10_1,
		D3D10_0,
	};


	bool initAPI(const API_Revision api_revision, const DXGI_FORMAT backBufferFmt, const DXGI_FORMAT depthBufferFmt, const int samples, const uint flags);
	bool createBuffers(const bool sampleBackBuffer);
	bool deleteBuffers();

	ID3D11Device *device;
	ID3D11DeviceContext *context;

	IDXGISwapChain *swapChain;

	ID3D11Texture2D *backBuffer;
	ID3D11Texture2D *depthBuffer;
	ID3D11RenderTargetView *backBufferRTV;
	ID3D11DepthStencilView *depthBufferDSV;
	TextureID backBufferTexture;

	DXGI_FORMAT backBufferFormat;
	DXGI_FORMAT depthBufferFormat;
	int msaaSamples;
};

#endif // _D3D11APP_H_
