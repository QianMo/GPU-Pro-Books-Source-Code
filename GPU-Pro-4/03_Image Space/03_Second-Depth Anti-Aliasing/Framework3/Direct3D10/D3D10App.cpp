
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

#include "D3D10App.h"

#pragma comment (lib, "dxgi.lib")
#ifdef USE_D3D10_1
#pragma comment (lib, "d3d10_1.lib")
#pragma comment (lib, "d3dx10.lib")
#else
#pragma comment (lib, "d3d10.lib")
#endif

D3D10App::D3D10App(){
	swapChain = NULL;
	device = NULL;

	backBuffer  = NULL;
	depthBuffer = NULL;
	backBufferRTV  = NULL;
	depthBufferDSV = NULL;
	backBufferFormat  = DXGI_FORMAT_UNKNOWN;
	depthBufferFormat = DXGI_FORMAT_UNKNOWN;
	backBufferTexture = TEXTURE_NONE;
}

bool D3D10App::initCaps(){
	renderer = NULL;
	return true;
}

bool D3D10App::initAPI(){
	DXGI_FORMAT colorFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	DXGI_FORMAT depthFormat = DXGI_FORMAT_UNKNOWN;
	switch (depthBits){
		case 16:
			depthFormat = DXGI_FORMAT_D16_UNORM;
			break;
		case 24:
			depthFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
			break;
		case 32:
			depthFormat = DXGI_FORMAT_D32_FLOAT;
			break;
	}

	return initAPI(colorFormat, depthFormat, max(antiAliasSamples, 1), 0);
}

void D3D10App::exitAPI(){
	deleteBuffers();

	delete renderer;

	if (swapChain){
		// Reset display mode to default
		if (fullscreen) swapChain->SetFullscreenState(false, NULL);
		swapChain->Release();
	}

	if (device){
		ULONG count = device->Release();
#ifdef _DEBUG
		if (count){
			char str[512];
			sprintf(str, "There are %d unreleased references left on the D3D device!\n", count);
			outputDebugString(str);
		}
#endif
		device = NULL;
	}

	DestroyWindow(hwnd);
}

bool D3D10App::initAPI(const DXGI_FORMAT backBufferFmt, const DXGI_FORMAT depthBufferFmt, const int samples, const uint flags){
	backBufferFormat = backBufferFmt;
	depthBufferFormat = depthBufferFmt;
	msaaSamples = samples;

	const bool sampleBackBuffer = (flags & SAMPLE_BACKBUFFER) != 0;

//	if (screen >= GetSystemMetrics(SM_CMONITORS)) screen = 0;

	IDXGIFactory *dxgiFactory;
	if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void **) &dxgiFactory))){
		ErrorMsg("Couldn't create DXGIFactory");
		return false;
	}

	IDXGIAdapter *dxgiAdapter;
	if (dxgiFactory->EnumAdapters(0, &dxgiAdapter) == DXGI_ERROR_NOT_FOUND){
		ErrorMsg("No adapters found");
		return false;
	}

//	DXGI_ADAPTER_DESC adapterDesc;
//	dxgiAdapter->GetDesc(&adapterDesc);

	IDXGIOutput *dxgiOutput;
	if (dxgiAdapter->EnumOutputs(0, &dxgiOutput) == DXGI_ERROR_NOT_FOUND){
		ErrorMsg("No outputs found");
		return false;
	}

	DXGI_OUTPUT_DESC oDesc;
	dxgiOutput->GetDesc(&oDesc);


	// Find a suitable fullscreen format
	int targetHz = 85;
	DXGI_RATIONAL fullScreenRefresh;
	int fsRefresh = 60;
	fullScreenRefresh.Numerator = fsRefresh;
	fullScreenRefresh.Denominator = 1;
	char str[128];

	uint nModes = 0;
	dxgiOutput->GetDisplayModeList(backBufferFormat, 0, &nModes, NULL);
	DXGI_MODE_DESC *modes = new DXGI_MODE_DESC[nModes];
	dxgiOutput->GetDisplayModeList(backBufferFormat, 0, &nModes, modes);

	resolution->clear();
	for (uint i = 0; i < nModes; i++){
		if (modes[i].Width >= 640 && modes[i].Height >= 480){
			sprintf(str, "%dx%d", modes[i].Width, modes[i].Height);
			int index = resolution->addItemUnique(str);

			if (int(modes[i].Width) == fullscreenWidth && int(modes[i].Height) == fullscreenHeight){
				int refresh = modes[i].RefreshRate.Numerator / modes[i].RefreshRate.Denominator;
				if (abs(refresh - targetHz) < abs(fsRefresh - targetHz)){
					fsRefresh = refresh;
					fullScreenRefresh = modes[i].RefreshRate;
				}
				resolution->selectItem(index);
			}
		}
	}
	delete [] modes;

	sprintf(str, "%s (%dx%d)", getTitle(), width, height);

	DWORD wndFlags = 0;
	int x, y, w, h;
	if (fullscreen){
		wndFlags |= WS_POPUP;
		x = y = 0;
		w = width;
		h = height;
	} else {
		wndFlags |= WS_OVERLAPPEDWINDOW;

		RECT wRect;
		wRect.left = 0;
		wRect.right = width;
		wRect.top = 0;
		wRect.bottom = height;
		AdjustWindowRect(&wRect, wndFlags, FALSE);

		MONITORINFO monInfo;
		monInfo.cbSize = sizeof(monInfo);
		GetMonitorInfo(oDesc.Monitor, &monInfo);

		w = min(wRect.right  - wRect.left, monInfo.rcWork.right  - monInfo.rcWork.left);
		h = min(wRect.bottom - wRect.top,  monInfo.rcWork.bottom - monInfo.rcWork.top);
		x = (monInfo.rcWork.left + monInfo.rcWork.right  - w) / 2;
		y = (monInfo.rcWork.top  + monInfo.rcWork.bottom - h) / 2;
	}


	hwnd = CreateWindow("Humus", str, wndFlags, x, y, w, h, HWND_DESKTOP, NULL, hInstance, NULL);

	RECT rect;
	GetClientRect(hwnd, &rect);

	// Create device and swap chain
	DXGI_SWAP_CHAIN_DESC sd;
	memset(&sd, 0, sizeof(sd));
	sd.BufferDesc.Width  = rect.right;
	sd.BufferDesc.Height = rect.bottom;
	sd.BufferDesc.Format = backBufferFormat;
	sd.BufferDesc.RefreshRate = fullScreenRefresh;
	sd.BufferUsage = /*DXGI_USAGE_BACK_BUFFER | */DXGI_USAGE_RENDER_TARGET_OUTPUT | ((sampleBackBuffer)? DXGI_USAGE_SHADER_INPUT : 0);
	sd.BufferCount = 1;
	sd.OutputWindow = hwnd;
	sd.Windowed = (BOOL) (!fullscreen);
	sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

	DWORD deviceFlags = D3D10_CREATE_DEVICE_SINGLETHREADED;
#ifdef _DEBUG
    deviceFlags |= D3D10_CREATE_DEVICE_DEBUG;
#endif

#ifdef USE_D3D10_1
	if (FAILED(D3D10CreateDevice1(dxgiAdapter, D3D10_DRIVER_TYPE_HARDWARE, NULL, deviceFlags, D3D10_FEATURE_LEVEL_10_1, D3D10_1_SDK_VERSION, &device))){
#else
	if (FAILED(D3D10CreateDevice(dxgiAdapter, D3D10_DRIVER_TYPE_HARDWARE, NULL, deviceFlags, D3D10_SDK_VERSION, &device))){
#endif
		ErrorMsg("Couldn't create D3D10 device");
		return false;
	}

//	HRESULT hr = device10_1->QueryInterface(__uuidof(ID3D10Device), (void **) &device);
//	HRESULT hr = device->QueryInterface(__uuidof(ID3D10Device1), (void **) &device10_1);

	while (msaaSamples > 0){
		UINT nQuality;
		if (SUCCEEDED(device->CheckMultisampleQualityLevels(backBufferFormat, msaaSamples, &nQuality)) && nQuality > 0){
			if ((flags & NO_SETTING_CHANGE) == 0) antiAliasSamples = msaaSamples;
			break;
		} else {
			msaaSamples -= 2;
		}
	}
	sd.SampleDesc.Count = msaaSamples;
	sd.SampleDesc.Quality = 0;

	if (FAILED(dxgiFactory->CreateSwapChain(device, &sd, &swapChain))){
		ErrorMsg("Couldn't create swapchain");
		return false;
	}

	// We'll handle Alt-Enter ourselves thank you very much ...
	dxgiFactory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER);

	dxgiOutput->Release();
	dxgiAdapter->Release();
	dxgiFactory->Release();

	if (fullscreen){
		captureMouse(!configDialog->isVisible());
	}

	renderer = new Direct3D10Renderer(device);

	if (!createBuffers(sampleBackBuffer)){
		delete renderer;
		return false;
	}
	antiAlias->selectItem(antiAliasSamples / 2);

	linearClamp = renderer->addSamplerState(LINEAR, CLAMP, CLAMP, CLAMP);
	defaultFont = renderer->addFont("../Textures/Fonts/Future.dds", "../Textures/Fonts/Future.font", linearClamp);
	blendSrcAlpha = renderer->addBlendState(SRC_ALPHA, ONE_MINUS_SRC_ALPHA);
	noDepthTest  = renderer->addDepthState(false, false);
	noDepthWrite = renderer->addDepthState(true,  false);
	cullNone  = renderer->addRasterizerState(CULL_NONE);
	cullBack  = renderer->addRasterizerState(CULL_BACK);
	cullFront = renderer->addRasterizerState(CULL_FRONT);

	return true;
}

bool D3D10App::createBuffers(const bool sampleBackBuffer){
	if (FAILED(swapChain->GetBuffer(0, __uuidof(ID3D10Texture2D), (LPVOID *) &backBuffer))) return false;

	if (FAILED(device->CreateRenderTargetView(backBuffer, NULL, &backBufferRTV))) return false;
	if (sampleBackBuffer){
		if ((backBufferTexture = ((Direct3D10Renderer *) renderer)->addTexture(backBuffer)) == TEXTURE_NONE) return false;
		backBuffer->AddRef();
	}


	if (depthBufferFormat != DXGI_FORMAT_UNKNOWN){
		// Create depth stencil texture
		D3D10_TEXTURE2D_DESC descDepth;
		descDepth.Width  = width;
		descDepth.Height = height;
		descDepth.MipLevels = 1;
		descDepth.ArraySize = 1;
		descDepth.Format = depthBufferFormat;
		descDepth.SampleDesc.Count = msaaSamples;
		descDepth.SampleDesc.Quality = 0;
		descDepth.Usage = D3D10_USAGE_DEFAULT;
		descDepth.BindFlags = D3D10_BIND_DEPTH_STENCIL;
		descDepth.CPUAccessFlags = 0;
		descDepth.MiscFlags = 0;
		if (FAILED(device->CreateTexture2D(&descDepth, NULL, &depthBuffer))){
			ErrorMsg("Couldn't create main depth buffer");
			return false;
		}

		// Create the depth stencil view
		D3D10_DEPTH_STENCIL_VIEW_DESC descDSV;
		descDSV.Format = descDepth.Format;
		descDSV.ViewDimension = msaaSamples > 1? D3D10_DSV_DIMENSION_TEXTURE2DMS : D3D10_DSV_DIMENSION_TEXTURE2D;
		descDSV.Texture2D.MipSlice = 0;
		if (FAILED(device->CreateDepthStencilView(depthBuffer, &descDSV, &depthBufferDSV))) return false;
	}

	device->OMSetRenderTargets(1, &backBufferRTV, depthBufferDSV);

	// Setup the viewport
	D3D10_VIEWPORT viewport;
	viewport.Width  = width;
	viewport.Height = height;
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	device->RSSetViewports(1, &viewport);

	((Direct3D10Renderer *) renderer)->setFrameBuffer(backBufferRTV, depthBufferDSV);

	return true;
}

bool D3D10App::deleteBuffers(){
	if (device) device->OMSetRenderTargets(0, NULL, NULL);

	if (backBufferTexture != TEXTURE_NONE){
		renderer->removeTexture(backBufferTexture);
		backBufferTexture = TEXTURE_NONE;
	}

	SAFE_RELEASE(backBuffer);
	SAFE_RELEASE(backBufferRTV);
	SAFE_RELEASE(depthBuffer);
	SAFE_RELEASE(depthBufferDSV);

	return true;
}

void D3D10App::beginFrame(){
	renderer->setViewport(width, height);
}

void D3D10App::endFrame(){
	swapChain->Present(vSync? 1 : 0, 0);
}

void D3D10App::onSize(const int w, const int h){
	BaseApp::onSize(w, h);

	if (device != NULL){
		const bool sampleBackBuffer = (backBufferTexture != TEXTURE_NONE);

		deleteBuffers();
		swapChain->ResizeBuffers(1, width, height, backBufferFormat, 0);
		createBuffers(sampleBackBuffer);
	}
}

bool D3D10App::captureScreenshot(Image &img){
	D3D10_TEXTURE2D_DESC desc;
	desc.Width = width;
	desc.Height = height;
	desc.Format = backBufferFormat;
	desc.ArraySize = 1;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.BindFlags = 0;
	desc.MipLevels = 1;
	desc.MiscFlags = 0;
	desc.Usage = D3D10_USAGE_STAGING;
	desc.CPUAccessFlags = D3D10_CPU_ACCESS_READ;

	bool result = false;

	ID3D10Texture2D *texture;
	if (SUCCEEDED(device->CreateTexture2D(&desc, NULL, &texture))){
		if (msaaSamples > 1){
			ID3D10Texture2D *resolved = NULL;
			desc.Usage = D3D10_USAGE_DEFAULT;
			desc.CPUAccessFlags = 0;

			if (SUCCEEDED(device->CreateTexture2D(&desc, NULL, &resolved))){
				device->ResolveSubresource(resolved, 0, backBuffer, 0, desc.Format);
				device->CopyResource(texture, resolved);
				resolved->Release();
			}
		} else {
			device->CopyResource(texture, backBuffer);
		}

		D3D10_MAPPED_TEXTURE2D map;
		if (SUCCEEDED(texture->Map(0, D3D10_MAP_READ, 0, &map))){
			if (backBufferFormat == DXGI_FORMAT_R10G10B10A2_UNORM){
				uint32 *dst = (uint32 *) img.create(FORMAT_RGB10A2, width, height, 1, 1);
				ubyte *src = (ubyte *) map.pData;

				for (int y = 0; y < height; y++){
					for (int x = 0; x < width; x++){
						dst[x] = ((uint32 *) src)[x] | 0xC0000000;
					}
					dst += width;
					src += map.RowPitch;
				}
			} else {
				ubyte *dst = img.create(FORMAT_RGB8, width, height, 1, 1);
				ubyte *src = (ubyte *) map.pData;

				for (int y = 0; y < height; y++){
					for (int x = 0; x < width; x++){
						dst[3 * x + 0] = src[4 * x + 0];
						dst[3 * x + 1] = src[4 * x + 1];
						dst[3 * x + 2] = src[4 * x + 2];
					}
					dst += width * 3;
					src += map.RowPitch;
				}
			}
			result = true;

			texture->Unmap(0);
		}

		texture->Release();
	}

	return result;
}
