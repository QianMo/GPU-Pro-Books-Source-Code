
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

#ifndef _D3D10APP_H_
#define _D3D10APP_H_

#include "Direct3D10Renderer.h"
#include "../BaseApp.h"

// InitAPI flags
#define NO_SETTING_CHANGE 0x1
#define SAMPLE_BACKBUFFER 0x2

class D3D10App : public BaseApp {
public:
	D3D10App();

	virtual bool initCaps();
	virtual bool initAPI();
	virtual void exitAPI();

	void beginFrame();
	void endFrame();

	virtual void onSize(const int w, const int h);

	bool captureScreenshot(Image &img);

protected:
	bool initAPI(const DXGI_FORMAT backBufferFmt, const DXGI_FORMAT depthBufferFmt, const int samples, const uint flags);
	bool createBuffers(const bool sampleBackBuffer);
	bool deleteBuffers();

#ifdef USE_D3D10_1
	ID3D10Device1 *device;
#else
	ID3D10Device *device;
#endif

	IDXGISwapChain *swapChain;

	ID3D10Texture2D *backBuffer;
	ID3D10Texture2D *depthBuffer;
	ID3D10RenderTargetView *backBufferRTV;
	ID3D10DepthStencilView *depthBufferDSV;
	TextureID backBufferTexture;

	DXGI_FORMAT backBufferFormat;
	DXGI_FORMAT depthBufferFormat;
	int msaaSamples;
};

#endif // _D3D10APP_H_
