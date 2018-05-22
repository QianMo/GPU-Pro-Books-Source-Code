#include "Core.h"

Core::Core()
{
	device = NULL;
	swapChain = NULL;
}

CoreResult Core::init(HWND wndMain, UINT swapChainBufferCount, D3D_DRIVER_TYPE driver, DXGI_FORMAT colorFormat, UINT refreshRateNumerator, 
				      UINT refreshRateDenominator, UINT sampleCount, UINT sampleQuality, bool windowed, 
				      IDXGIAdapter* adapter)
{
	RECT wndRect;
    GetClientRect(wndMain, &wndRect);
    UINT wndWidth = wndRect.right - wndRect.left;
    UINT wndHeight = wndRect.bottom - wndRect.top;
	renderTargetView = NULL;
	renderTargetViewOverwrite = NULL;
	depthStencilView = NULL;
	backBuffer = NULL;
	depthStencil = NULL;
	depthStencilSRV = NULL;

	DXGI_SWAP_CHAIN_DESC swapChainDesc;

	CoreLog::Information(L"Initializing Direct3D11.");

	// Fill the swapChainDesc
    ZeroMemory(&swapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
	swapChainDesc.BufferCount = swapChainBufferCount;
    swapChainDesc.BufferDesc.Width = wndWidth;
    swapChainDesc.BufferDesc.Height = wndHeight;
    swapChainDesc.BufferDesc.Format = colorFormat;
    swapChainDesc.BufferDesc.RefreshRate.Numerator = refreshRateNumerator;
    swapChainDesc.BufferDesc.RefreshRate.Denominator = refreshRateDenominator;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.OutputWindow = wndMain;
    swapChainDesc.SampleDesc.Count = sampleCount;
    swapChainDesc.SampleDesc.Quality = sampleQuality;
    swapChainDesc.Windowed = windowed;
	swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH; // Automatically switch resolution when in fullscreen mode

	// Create SwapChain and Device
	HRESULT result;
	D3D_FEATURE_LEVEL featureLevels[1] = { D3D_FEATURE_LEVEL_11_0 };
#ifndef _DEBUG		
	result = D3D11CreateDeviceAndSwapChain(adapter, driver, NULL, 0, featureLevels, 1, D3D11_SDK_VERSION, &swapChainDesc, &swapChain, &device, NULL, &immediateDeviceContext);
#else
	result = D3D11CreateDeviceAndSwapChain(adapter, driver, NULL, D3D11_CREATE_DEVICE_DEBUG, featureLevels, 1, D3D11_SDK_VERSION, &swapChainDesc, &swapChain, &device, NULL, &immediateDeviceContext);
#endif

	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create the device and swap chain!, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}

	ID3D11Texture2D* backBufferTex;
	
	result = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void **) &backBufferTex);
	
	if(FAILED(result))
	{
		device->Release();
		swapChain->Release();
		CoreLog::Information(L"Could not get the backbuffer from the swap chain!, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}

	CoreResult cresult = CreateTexture2D(backBufferTex, &backBuffer);
	if(cresult != CORE_OK)
	{
		swapChain->Release();
		return cresult;
	}
	backBufferTex->Release();
	

	cresult = backBuffer->CreateRenderTargetView(NULL, &renderTargetView);
	if(cresult != CORE_OK)
	{
		device->Release();
		swapChain->Release();
		backBuffer->Release();
		return cresult;
	}
	renderTargetViewOverwrite = renderTargetView;

	// Create depth stencil texture
	CreateTexture2D(NULL, wndWidth, wndHeight, 1, 1, DXGI_FORMAT_R32_TYPELESS, 0, 0, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL, 1, 0, &depthStencil); 
	
	D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
    dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    dsvDesc.Texture2D.MipSlice = 0;
	dsvDesc.Flags = 0;
	
	// Create the depth stencil view
	cresult = depthStencil->CreateDepthStencilView(&dsvDesc, &depthStencilView);
    if(cresult != CORE_OK)
	{
		device->Release();
		swapChain->Release();
		backBuffer->Release();
		renderTargetView->Release();
		return result;
	}
	depthStencilViewOverwrite = depthStencilView;

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;

	srvDesc.Texture2D.MipLevels = 1;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
	
	cresult = depthStencil->CreateShaderResourceView(&srvDesc, &depthStencilSRV);

	if(cresult != CORE_OK)
	{
		device->Release();
		swapChain->Release();
		backBuffer->Release();
		renderTargetView->Release();
		depthStencilView->Release();
		return result;
	}

	// Set the views
	immediateDeviceContext->OMSetRenderTargets(1, &renderTargetView, depthStencilView);
	
	// Optimize math with 3DNow! Prof if available (only x86)
	#ifndef WIN64
	if(ProcessorSupports3DNowProf())
	{
		CoreLog::Information(L"Using 3DNow Professional vector math.");
		
		// CoreVector3
		CoreVector3Length = &CoreVector3Length_3DNow;
		CoreVector3LengthSq = &CoreVector3LengthSq_3DNow;
		CoreVector3Cross = &CoreVector3Cross_3DNow;
		CoreVector3Dot = &CoreVector3Dot_3DNow;
		CoreVector3Normalize = &CoreVector3Normalize_3DNow;
		CoreVector3TransformCoords = &CoreVector3TransformCoords_3DNow;
		CoreVector3TransformNormal = &CoreVector3TransformNormal_3DNow;

		// CoreMatrix4x4
		CoreMatrix4x4Add = &CoreMatrix4x4Add_3DNow;
		CoreMatrix4x4Sub = &CoreMatrix4x4Sub_3DNow;
		CoreMatrix4x4Mul = &CoreMatrix4x4Mul_3DNow;
		CoreMatrix4x4MulFloat = &CoreMatrix4x4MulFloat_3DNow;
		CoreMatrix4x4DivFloat = &CoreMatrix4x4DivFloat_3DNow;
	}
	#endif
	
	return CORE_OK;
}

// Create a D3D device without SwapChain
CoreResult Core::init(D3D_DRIVER_TYPE driver, IDXGIAdapter* adapter)
{
	depthStencilSRV = NULL;
	CoreLog::Information(L"Initializing Direct3D11.");

	// Create SwapChain and Device
	HRESULT result;
	D3D_FEATURE_LEVEL featureLevels[1] = { D3D_FEATURE_LEVEL_11_0 };
#ifndef _DEBUG						// SOFTWARE Mode is not suitable for any application; On Debug mode compile both Devices are created
	result = D3D11CreateDevice(adapter, driver, NULL, 0, featureLevels, 1, D3D11_SDK_VERSION, &device, NULL, &immediateDeviceContext);
#else
	result = D3D11CreateDevice(adapter, driver, NULL, D3D11_CREATE_DEVICE_DEBUG, featureLevels, 1, D3D11_SDK_VERSION, &device, NULL, &immediateDeviceContext);
#endif

	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create the device and swap chain!, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	
	// Optimize math with 3DNow! Prof if available (only x86)
	#ifndef WIN64
	if(ProcessorSupports3DNowProf())
	{
		CoreLog::Information(L"Using 3DNow Professional vector math.");
		
		// CoreVector3
		CoreVector3Length = &CoreVector3Length_3DNow;
		CoreVector3LengthSq = &CoreVector3LengthSq_3DNow;
		CoreVector3Cross = &CoreVector3Cross_3DNow;
		CoreVector3Dot = &CoreVector3Dot_3DNow;
		CoreVector3Normalize = &CoreVector3Normalize_3DNow;
		CoreVector3TransformCoords = &CoreVector3TransformCoords_3DNow;
		CoreVector3TransformNormal = &CoreVector3TransformNormal_3DNow;

		// CoreMatrix4x4
		CoreMatrix4x4Add = &CoreMatrix4x4Add_3DNow;
		CoreMatrix4x4Sub = &CoreMatrix4x4Sub_3DNow;
		CoreMatrix4x4Mul = &CoreMatrix4x4Mul_3DNow;
		CoreMatrix4x4MulFloat = &CoreMatrix4x4MulFloat_3DNow;
		CoreMatrix4x4DivFloat = &CoreMatrix4x4DivFloat_3DNow;
	}
	#endif

	renderTargetView = NULL;
	depthStencilView = NULL;
	backBuffer = NULL;
	depthStencil = NULL;
	swapChain = NULL;
	
	return CORE_OK;
}

CoreResult Core::CreateTexture2D(const std::vector <std::istream *> &in, UINT mipLevels,
				  		   UINT cpuAccessFlag, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
						   UINT sampleCount, UINT sampleQuality, bool sRGB, CoreTexture2D** outTex)
{
	if(!outTex) return CORE_INVALIDARGS;

	*outTex = new CoreTexture2D();
	
	if(!*outTex) return CORE_OUTOFMEM;
	
	CoreResult result = (*outTex)->init(this, in, mipLevels, cpuAccessFlag, miscFlags, usage, bindFlags, sampleCount, sampleQuality, sRGB);
	if(result != CORE_OK)
	{
		(*outTex)->Release();
		(*outTex) = NULL;
		return result;
	}

	return CORE_OK;
}
// Load a texture from a stream
CoreResult Core::CreateTexture2D(std::istream *in[], UINT textureCount, UINT mipLevels,
				  		   UINT cpuAccessFlag, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
						   UINT sampleCount, UINT sampleQuality, bool sRGB, CoreTexture2D** outTex)
{
	if(!outTex) return CORE_INVALIDARGS;

	*outTex = new CoreTexture2D();
	
	if(!*outTex) return CORE_OUTOFMEM;
	
	CoreResult result = (*outTex)->init(this, in, textureCount, mipLevels, cpuAccessFlag, miscFlags, usage, bindFlags, sampleCount, sampleQuality,sRGB);
	if(result != CORE_OK)
	{
		(*outTex)->Release();
		(*outTex) = NULL;
		return result;
	}

	return CORE_OK;
}

// Load a texture from a stream
CoreResult Core::CreateTexture2D(std::istream &in, UINT mipLevels,
				  		   UINT cpuAccessFlag, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
						   UINT sampleCount, UINT sampleQuality, bool sRGB, CoreTexture2D** outTex)
{
	if(!outTex) return CORE_INVALIDARGS;

	std::istream *dummy[1];
	dummy[0] = &in;

	*outTex = new CoreTexture2D();
	
	if(!*outTex) return CORE_OUTOFMEM;
	
	CoreResult result = (*outTex)->init(this, dummy, 1, mipLevels, cpuAccessFlag, miscFlags, usage, bindFlags, sampleCount, sampleQuality, sRGB);
	if(result != CORE_OK)
	{
		(*outTex)->Release();
		(*outTex) = NULL;
		return result;
	}

	return CORE_OK;
}

// Create a texture from memory
CoreResult Core::CreateTexture2D(BYTE** data, UINT width, UINT height, UINT textureCount, UINT mipLevels,
							     DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
							     UINT sampleCount, UINT sampleQuality, CoreTexture2D** outTex)
{
	if(!outTex) return CORE_INVALIDARGS;

	*outTex = new CoreTexture2D();
	
	if(!*outTex) return CORE_OUTOFMEM;
	
	CoreResult result = (*outTex)->init(this, data, width, height, textureCount, mipLevels, format, cpuAccessFlags, miscFlags, usage, bindFlags, sampleCount, sampleQuality);
	if(result != CORE_OK)
	{
		(*outTex)->Release();
		(*outTex) = NULL;
		return result;
	}

	return CORE_OK;
}

// Directly use an ID3D11Texture2D object
CoreResult Core::CreateTexture2D(ID3D11Texture2D* texture, CoreTexture2D** outTex)
{
	if(!outTex) return CORE_INVALIDARGS;

	*outTex = new CoreTexture2D();
	
	if(!*outTex) return CORE_OUTOFMEM;
	
	CoreResult result = (*outTex)->init(this, texture);
	if(result != CORE_OK)
	{
		(*outTex)->Release();
		(*outTex) = NULL;
		return result;
	}

	return CORE_OK;
}

// Create an additional SwapChain
CoreResult Core::CreateSwapChain(HWND wndMain, UINT swapChainBufferCount, DXGI_FORMAT colorFormat, DXGI_FORMAT depthStencilFormat, UINT refreshRateNumerator, 
								 UINT refreshRateDenominator, UINT sampleCount, UINT sampleQuality, bool windowed,
								 IDXGISwapChain** swapChain, CoreTexture2D** backBuffer, ID3D11RenderTargetView** renderTargetView, 
								 CoreTexture2D** depthStencil, ID3D11DepthStencilView** depthStencilView)
{
	DXGI_SWAP_CHAIN_DESC swapChainDesc;

	RECT wndRect;
    GetClientRect(wndMain, &wndRect);
    UINT wndWidth = wndRect.right - wndRect.left;
    UINT wndHeight = wndRect.bottom - wndRect.top;

	// Fill the swapChainDesc
    ZeroMemory(&swapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
	swapChainDesc.BufferCount = swapChainBufferCount;
    swapChainDesc.BufferDesc.Width = wndWidth;
    swapChainDesc.BufferDesc.Height = wndHeight;
    swapChainDesc.BufferDesc.Format = colorFormat;
    swapChainDesc.BufferDesc.RefreshRate.Numerator = refreshRateNumerator;
    swapChainDesc.BufferDesc.RefreshRate.Denominator = refreshRateDenominator;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.OutputWindow = wndMain;
    swapChainDesc.SampleDesc.Count = sampleCount;
    swapChainDesc.SampleDesc.Quality = sampleQuality;
    swapChainDesc.Windowed = windowed;
	swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH; // Automatically switch resolution when in fullscreen mode

	IDXGIFactory* factory;
	HRESULT result = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)(&factory) );
	if(FAILED(result))
		return CORE_MISC_ERROR;

	factory->CreateSwapChain(device, &swapChainDesc, swapChain);
	factory->Release();
	
	ID3D11Texture2D* backBufferTex;
	
	result = (*swapChain)->GetBuffer(0, __uuidof(ID3D11Texture2D), (void **) &backBufferTex);
	
	if(FAILED(result))
	{
		(*swapChain)->Release();
		CoreLog::Information(L"Could not get the backbuffer from the swap chain!, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}

	CoreResult cresult = CreateTexture2D(backBufferTex, backBuffer);
	if(cresult != CORE_OK)
	{
		(*swapChain)->Release();
		return cresult;
	}
	
	backBufferTex->Release();
	

	cresult = (*backBuffer)->CreateRenderTargetView(NULL, renderTargetView);
	if(cresult != CORE_OK)
	{
		(*swapChain)->Release();
		(*backBuffer)->Release();
		return cresult;
	}

	// Create depth stencil texture
	CreateTexture2D(NULL, wndWidth, wndHeight, 1, 1, depthStencilFormat, 0, 0, D3D11_USAGE_DEFAULT, D3D11_BIND_DEPTH_STENCIL, 1, 0, depthStencil); 
	
	D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
    dsvDesc.Format = (*depthStencil)->GetFormat();
    dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    dsvDesc.Texture2D.MipSlice = 0;
	dsvDesc.Flags = 0;
	
	// Create the depth stencil view
	cresult = (*depthStencil)->CreateDepthStencilView(&dsvDesc, depthStencilView);
    if(cresult != CORE_OK)
	{
		(*swapChain)->Release();
		(*backBuffer)->Release();
		(*renderTargetView)->Release();
		return result;
	}

	return CORE_OK;

}

// Resizes an additionally created SwapChain
CoreResult Core::ResizeSwapChain(UINT width, UINT height, DXGI_FORMAT format, DXGI_FORMAT depthStencilFormat, IDXGISwapChain* swapChain, CoreTexture2D** backBuffer, ID3D11RenderTargetView** renderTargetView, 
								 CoreTexture2D** depthStencil, ID3D11DepthStencilView** depthStencilView)
{	
	DXGI_SWAP_CHAIN_DESC desc;
	swapChain->GetDesc(&desc);

	(*renderTargetView)->Release();
	(*renderTargetView) = NULL;
	(*depthStencilView)->Release();
	(*depthStencilView) = NULL;
	(*backBuffer)->Release();
	(*backBuffer) = NULL;
	(*depthStencil)->Release();
	(*depthStencil) = NULL;

	HRESULT result = swapChain->ResizeBuffers(1, width, height, format, desc.Flags);
	if(FAILED(result))
		return CORE_MISC_ERROR;

	
	ID3D11Texture2D* backBufferTex;
	
	result = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void **) &backBufferTex);
	
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not get the backbuffer from the swap chain!, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}

	CoreResult cresult = CreateTexture2D(backBufferTex, backBuffer);
	if(cresult != CORE_OK)
		return cresult;
	
	backBufferTex->Release();
	

	cresult = (*backBuffer)->CreateRenderTargetView(NULL, renderTargetView);
	if(cresult != CORE_OK)
	{
		(*backBuffer)->Release();
		return cresult;
	}

	// Create depth stencil texture
	CreateTexture2D(NULL, width, height, 1, 1, depthStencilFormat, 0, 0, D3D11_USAGE_DEFAULT, D3D11_BIND_DEPTH_STENCIL, 1, 0, depthStencil); 
	
	D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
    dsvDesc.Format = (*depthStencil)->GetFormat();
    dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    dsvDesc.Texture2D.MipSlice = 0;
	dsvDesc.Flags = 0;
	
	// Create the depth stencil view
	cresult = (*depthStencil)->CreateDepthStencilView(&dsvDesc, depthStencilView);
    if(cresult != CORE_OK)
	{
		(*backBuffer)->Release();
		(*renderTargetView)->Release();
		return result;
	}

	return CORE_OK;
}

// Resizes the internal SwapChain
CoreResult Core::ResizeSwapChain(UINT width, UINT height, DXGI_FORMAT format)
{
	DXGI_SWAP_CHAIN_DESC desc;
	swapChain->GetDesc(&desc);

	renderTargetView->Release();
	renderTargetView = NULL;
	renderTargetViewOverwrite = NULL;
	depthStencilView->Release();
	depthStencilView = NULL;
	backBuffer->Release();
	backBuffer = NULL;
	depthStencilSRV->Release();
	depthStencilSRV = NULL;
	depthStencil->Release();
	depthStencil = NULL;

	SetRenderTargets(0, NULL, NULL);

	HRESULT result = swapChain->ResizeBuffers(1, width, height, format, desc.Flags);
	if(FAILED(result))
	{
		CoreLog::Information(L"Backbuffer resize error!, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	
	ID3D11Texture2D* backBufferTex;
	
	result = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void **) &backBufferTex);
	
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not get the backbuffer from the swap chain!, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}

	CoreResult cresult = CreateTexture2D(backBufferTex, &backBuffer);
	if(cresult != CORE_OK)
		return cresult;
	
	backBufferTex->Release();
	
	cresult = backBuffer->CreateRenderTargetView(NULL, &renderTargetView);
	if(cresult != CORE_OK)
	{
		backBuffer->Release();
		return cresult;
	}
	renderTargetViewOverwrite = renderTargetView;

	// Create depth stencil texture
	CreateTexture2D(NULL, width, height, 1, 1, DXGI_FORMAT_R32_TYPELESS, 0, 0, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL, 1, 0, &depthStencil); 
	
	D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
    dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    dsvDesc.Texture2D.MipSlice = 0;
	dsvDesc.Flags = 0;
	
	// Create the depth stencil view
	cresult = depthStencil->CreateDepthStencilView(&dsvDesc, &depthStencilView);
    if(cresult != CORE_OK)
	{
		backBuffer->Release();
		renderTargetView->Release();
		return result;
	}

	depthStencilViewOverwrite = depthStencilView;

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;

	srvDesc.Texture2D.MipLevels = 1;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
	
	cresult = depthStencil->CreateShaderResourceView(&srvDesc, &depthStencilSRV);

	if(cresult != CORE_OK)
	{
		backBuffer->Release();
		renderTargetView->Release();
		depthStencilView->Release();
		return result;
	}

	SetDefaultRenderTarget();
	return CORE_OK;
}

void Core::finalRelease()
{
	if(immediateDeviceContext) immediateDeviceContext->OMSetRenderTargets(0, NULL, NULL);
	SAFE_RELEASE(renderTargetView);
	SAFE_RELEASE(depthStencilView);
	SAFE_RELEASE(backBuffer);
	SAFE_RELEASE(depthStencil);
	if(swapChain)
		swapChain->SetFullscreenState(false, NULL);
	SAFE_RELEASE(swapChain);
	SAFE_RELEASE(device);
	std::vector<D3D11_VIEWPORT*>::iterator it = savedViewportsStack.begin();
	while(it != savedViewportsStack.end())
	{
		SAFE_DELETE(*it);
		it++;
	}
}

// Creates the Core
CoreResult CreateCore(HWND wndMain, UINT swapChainBufferCount, D3D_DRIVER_TYPE driver, DXGI_FORMAT colorFormat,
					  UINT refreshRateNumerator, UINT refreshRateDenominator, UINT sampleCount, 
					  UINT sampleQuality, bool windowed, IDXGIAdapter* adapter, Core** outCore)
{
	if(!outCore) return CORE_INVALIDARGS;

	*outCore = new Core();
	
	if(!*outCore) return CORE_OUTOFMEM;
	
	CoreResult result = (*outCore)->init(wndMain, swapChainBufferCount, driver, colorFormat, refreshRateNumerator, refreshRateDenominator, sampleCount, sampleQuality, windowed, adapter);
	if(result != CORE_OK)
	{
		(*outCore)->Release();
		(*outCore) = NULL;
		return result;
	}

	return CORE_OK;
}

// Create a D3D device without SwapChain
CoreResult CreateCore(D3D_DRIVER_TYPE driver, IDXGIAdapter* adapter, Core** outCore)
{
	if(!outCore) return CORE_INVALIDARGS;

	*outCore = new Core();
	
	if(!*outCore) return CORE_OUTOFMEM;
	
	CoreResult result = (*outCore)->init(driver, adapter);
	if(result != CORE_OK)
	{
		(*outCore)->Release();
		(*outCore) = NULL;
		return result;
	}

	return CORE_OK;
}

// Load a 3D texture from a stream
CoreResult Core::CreateTexture3D(std::istream& in, UINT mipLevels,
			  	    UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags, CoreTexture3D** outTex)
{
	if(!outTex) return CORE_INVALIDARGS;

	*outTex = new CoreTexture3D();
	
	if(!*outTex) return CORE_OUTOFMEM;
	
	CoreResult result = (*outTex)->init(this, in, mipLevels, cpuAccessFlags, miscFlags, usage, bindFlags);
	if(result != CORE_OK)
	{
		(*outTex)->Release();
		(*outTex) = NULL;
		return result;
	}

	return CORE_OK;
}

// Create a texture from memory
CoreResult Core::CreateTexture1D(BYTE** data, UINT width, UINT textureCount, UINT mipLevels,
							   DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
							   CoreTexture1D** outTex) {
	if(!outTex) return CORE_INVALIDARGS;

	*outTex = new CoreTexture1D();
	
	if(!*outTex) return CORE_OUTOFMEM;
	
	CoreResult result = (*outTex)->init(this, data, width, textureCount, mipLevels, format, cpuAccessFlags, miscFlags, usage, bindFlags);
	if(result != CORE_OK)
	{
		(*outTex)->Release();
		(*outTex) = NULL;
		return result;
	}

	return CORE_OK;
}

// Load a 3D texture from memory
CoreResult Core::CreateTexture3D(BYTE* data, UINT width, UINT height, UINT depth, UINT mipLevels,
		  	    DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
				CoreTexture3D** outTex)
{
	if(!outTex) return CORE_INVALIDARGS;

	*outTex = new CoreTexture3D();
	
	if(!*outTex) return CORE_OUTOFMEM;
	
	CoreResult result = (*outTex)->init(this, data, width, height, depth, mipLevels, format, cpuAccessFlags, miscFlags, usage, bindFlags);
	if(result != CORE_OK)
	{
		(*outTex)->Release();
		(*outTex) = NULL;
		return result;
	}

	return CORE_OK;
}

// Clear the capsuled RenderTargetView
void Core::ClearRenderTargetView(CoreColor &c)
{
	immediateDeviceContext->ClearRenderTargetView(renderTargetViewOverwrite, (float *)&c);
}

// Clear a different RenderTargetView
void Core::ClearRenderTargetView(ID3D11RenderTargetView* rtv, CoreColor &c)
{
	immediateDeviceContext->ClearRenderTargetView(rtv, (float *)&c);
}

// Clear the capsuled DepthStencilView
void Core::ClearDepthStencilView(UINT clearFlags, float depth, UINT8 stencil)
{
	immediateDeviceContext->ClearDepthStencilView(depthStencilViewOverwrite, clearFlags, depth, stencil);
}

// Sets the internal render target/ depth stencil view
void Core::SetDefaultRenderTarget()
{
	immediateDeviceContext->OMSetRenderTargets(1, &renderTargetViewOverwrite, depthStencilViewOverwrite);
}

void Core::OverwriteDefaultRenderTargetView(ID3D11RenderTargetView *renderTarget)
{
	if(renderTargetViewOverwrite != renderTarget)	// Push to stack if overridden more times
		overriddenRenderTargetStack.push_back(renderTargetViewOverwrite);
	renderTargetViewOverwrite = renderTarget;
}

void Core::RestoreDefaultRenderTargetView()
{
	if(overriddenRenderTargetStack.size() == 0)
		renderTargetViewOverwrite = renderTargetView;
	else
	{
		renderTargetViewOverwrite = overriddenRenderTargetStack.back();
		overriddenRenderTargetStack.pop_back();
	}
}

void Core::OverwriteDefaultDepthStencilView(ID3D11DepthStencilView *dsv)
{
	if(depthStencilViewOverwrite != depthStencilView)	// Push to stack if overridden more times
		overriddenDepthStencilStack.push_back(depthStencilViewOverwrite);
	depthStencilViewOverwrite = dsv;
}

void Core::RestoreDefaultDepthStencilView()
{
	if(overriddenDepthStencilStack.size() == 0)
		depthStencilViewOverwrite = depthStencilView;
	else
	{
		depthStencilViewOverwrite = overriddenDepthStencilStack.back();
		overriddenDepthStencilStack.pop_back();
	}
}

// Sets the RenderTargets
void Core::SetRenderTargets(UINT numRenderTargets, ID3D11RenderTargetView** rtvs, ID3D11DepthStencilView* dsv)
{
	immediateDeviceContext->OMSetRenderTargets(numRenderTargets, rtvs, dsv);
}

// Saves all Viewports -> restore with RestoreViewports
void Core::SaveViewports()
{
	D3D11_VIEWPORT *savedViewports;
	UINT numSavedViewports = 0;
	immediateDeviceContext->RSGetViewports(&numSavedViewports, NULL);
	savedViewports = new D3D11_VIEWPORT[numSavedViewports];
	immediateDeviceContext->RSGetViewports(&numSavedViewports, savedViewports);
	savedViewportsStack.push_back(savedViewports);
	numSavedViewportsStack.push_back(numSavedViewports);
}

// Restores all Viewports
void Core::RestoreViewports()
{
	if(savedViewportsStack.size() > 0)
	{
		D3D11_VIEWPORT *savedViewports = savedViewportsStack.back();
		savedViewportsStack.pop_back();
		UINT numSavedViewports = numSavedViewportsStack.back();
		numSavedViewportsStack.pop_back();
		immediateDeviceContext->RSSetViewports(numSavedViewports, savedViewports);
		SAFE_DELETE(savedViewports);
	}
	else
		CoreLog::Information("Viewport restore without any save before!");
}