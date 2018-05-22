
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

#include "D3DApp.h"

#pragma comment (lib, "d3d9.lib")
#pragma comment (lib, "d3dx9.lib")

D3DApp::D3DApp(){
	d3d = NULL;
	dev = NULL;
}

bool D3DApp::initCaps(){
	renderer = NULL;

	if ((d3d = Direct3DCreate9(D3D_SDK_VERSION)) == NULL){
		MessageBox(hwnd, "Couldn't initialize Direct3D\nMake sure you have DirectX 9.0c or later installed.", "Error", MB_OK | MB_ICONERROR);
		return false;
	}

	d3d->GetDeviceCaps(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, &caps);
	return true;
}

bool D3DApp::initAPI(){
	if (screen >= GetSystemMetrics(SM_CMONITORS)) screen = 0;

	memset(&d3dpp, 0, sizeof(d3dpp));

	d3dpp.BackBufferFormat = D3DFMT_X8R8G8B8;
	d3dpp.Windowed = !fullscreen;


	// Find a suitable fullscreen format
	int fullScreenRefresh = 60, targetHz = 85;
	char str[128];

	int nModes = d3d->GetAdapterModeCount(D3DADAPTER_DEFAULT, D3DFMT_X8R8G8B8);

	resolution->clear();
	for (int i = 0; i < nModes; i++){
		D3DDISPLAYMODE mode;
		d3d->EnumAdapterModes(D3DADAPTER_DEFAULT, D3DFMT_X8R8G8B8, i, &mode);

		if (mode.Width >= 640 && mode.Height >= 480){
			sprintf(str, "%dx%d", mode.Width, mode.Height);
			int index = resolution->addItemUnique(str);

			if (int(mode.Width) == fullscreenWidth && int(mode.Height) == fullscreenHeight){
				if (abs(int(mode.RefreshRate) - targetHz) < abs(fullScreenRefresh - targetHz)){
					fullScreenRefresh = mode.RefreshRate;
				}
				resolution->selectItem(index);
			}
		}

	}

	sprintf(str, "%s (%dx%d)", getTitle(), width, height);

	DWORD flags = 0;
	int x, y, w, h;
	if (fullscreen){
		flags |= WS_POPUP;
		x = y = 0;
		w = width;
		h = height;

		d3dpp.FullScreen_RefreshRateInHz = fullScreenRefresh;
	
	} else {
		flags |= WS_OVERLAPPEDWINDOW;

		RECT wRect;
		wRect.left = 0;
		wRect.right = width;
		wRect.top = 0;
		wRect.bottom = height;
		AdjustWindowRect(&wRect, flags, FALSE);

		HMONITOR hMonitor = d3d->GetAdapterMonitor(D3DADAPTER_DEFAULT);

		MONITORINFO monInfo;
		monInfo.cbSize = sizeof(monInfo);
		GetMonitorInfo(hMonitor, &monInfo);

		w = min(wRect.right  - wRect.left, monInfo.rcWork.right  - monInfo.rcWork.left);
		h = min(wRect.bottom - wRect.top,  monInfo.rcWork.bottom - monInfo.rcWork.top);

		x = (monInfo.rcWork.left + monInfo.rcWork.right  - w) / 2;
		y = (monInfo.rcWork.top  + monInfo.rcWork.bottom - h) / 2;
	}


	hwnd = CreateWindow("Humus", str, flags, x, y, w, h, HWND_DESKTOP, NULL, hInstance, NULL);


	d3dpp.BackBufferWidth  = width;
	d3dpp.BackBufferHeight = height;
	d3dpp.BackBufferCount  = 1;

	d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
	d3dpp.SwapEffect           = D3DSWAPEFFECT_DISCARD;

	d3dpp.EnableAutoDepthStencil = (depthBits > 0);
	d3dpp.AutoDepthStencilFormat = (depthBits > 16)? ((stencilBits > 0)? D3DFMT_D24S8 : D3DFMT_D24X8) : D3DFMT_D16;

	int multiSample = antiAliasSamples;
	while (true){
		d3dpp.MultiSampleType = (D3DMULTISAMPLE_TYPE) multiSample;

		if (d3d->CreateDevice(screen, D3DDEVTYPE_HAL, hwnd, /*D3DCREATE_PUREDEVICE | */ D3DCREATE_HARDWARE_VERTEXPROCESSING, &d3dpp, &dev) == D3D_OK){
			antiAliasSamples = multiSample;
			break;
		} else {
			if (multiSample > 0){
				multiSample -= 2;
			} else {
				MessageBox(hwnd, "Couldn't create Direct3D device interface.", "Error", MB_OK | MB_ICONERROR);
				return false;
			}
		}


	}

	if (fullscreen){
		captureMouse(!configDialog->isVisible());
	}


	renderer = new Direct3DRenderer(dev, caps);

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

void D3DApp::exitAPI(){
	delete renderer;

    if (dev != NULL){
		dev->Release();
		dev = NULL;
	}

    if (d3d != NULL){
		d3d->Release();
		d3d = NULL;
	}


	DestroyWindow(hwnd);

	if (fullscreen){
		// Reset display mode to default
		ChangeDisplaySettingsEx(device.DeviceName, NULL, NULL, 0, NULL);
	}
}

void D3DApp::beginFrame(){
	renderer->setViewport(width, height);
	dev->BeginScene();
}

void D3DApp::endFrame(){
    dev->EndScene();
    dev->Present(NULL, NULL, NULL, NULL);
}

void D3DApp::onSize(const int w, const int h){
	BaseApp::onSize(w, h);

	if (dev != NULL){
		d3dpp.BackBufferWidth  = w;
		d3dpp.BackBufferHeight = h;

		((Direct3DRenderer *) renderer)->resetDevice(d3dpp);

		D3DVIEWPORT9 vp = { 0, 0, w, h, 0.0f, 1.0f };
		dev->SetViewport(&vp);

		onReset();
	}
}

bool D3DApp::captureScreenshot(Image &img){
	D3DDISPLAYMODE dmode;
	d3d->GetAdapterDisplayMode(D3DADAPTER_DEFAULT, &dmode);

	POINT topLeft = { 0, 0 };
	ClientToScreen(hwnd, &topLeft);

	bool result = false;

	LPDIRECT3DSURFACE9 surface;
	if (dev->CreateOffscreenPlainSurface(dmode.Width, dmode.Height, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH, &surface, NULL) == D3D_OK){
		if (dev->GetFrontBufferData(0, surface) == D3D_OK){
			D3DLOCKED_RECT lockedRect;
			if (surface->LockRect(&lockedRect, NULL, D3DLOCK_READONLY) == D3D_OK){

				ubyte *dst = img.create(FORMAT_RGB8, width, height, 1, 1);
				for (int y = 0; y < height; y++){
					ubyte *src = ((ubyte *) lockedRect.pBits) + 4 * ((topLeft.y + y) * dmode.Width + topLeft.x);
					for (int x = 0; x < width; x++){
						dst[0] = src[2];
						dst[1] = src[1];
						dst[2] = src[0];
						dst += 3;
						src += 4;
					}
				}

				surface->UnlockRect();
				result = true;
			}
		}

		surface->Release();
	}

	return result;
}
