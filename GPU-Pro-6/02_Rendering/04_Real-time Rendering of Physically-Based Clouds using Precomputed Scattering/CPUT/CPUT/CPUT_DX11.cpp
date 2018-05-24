//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------
#include "CPUT_DX11.h"
#include "CPUTRenderStateBlockDX11.h"
#include "CPUTBufferDX11.h"
#include "CPUTTextureDX11.h"

// static initializers
ID3D11Device* CPUT_DX11::mpD3dDevice = NULL;
CPUT_DX11 *gpSample;

// Destructor
//-----------------------------------------------------------------------------
CPUT_DX11::~CPUT_DX11()
{
    // all previous shutdown tasks should have happened in CPUTShutdown()

    // We created the default renderstate block, we release it.
    
    CPUTRenderStateBlock *pRenderState = CPUTRenderStateBlock::GetDefaultRenderStateBlock();
    SAFE_RELEASE(pRenderState);
    SAFE_RELEASE(mpPerFrameConstantBuffer);
    SAFE_RELEASE(mpBackBufferSRV);
    SAFE_RELEASE(mpBackBufferUAV);
    SAFE_RELEASE(mpBackBuffer);
    SAFE_RELEASE(mpDepthBuffer);
	SAFE_RELEASE(mpBackBufferTexture);
    SAFE_RELEASE(mpDepthBufferTexture);


    SAFE_RELEASE(mpDepthStencilSRV);

    // destroy the window
    if(mpWindow)
    {
        delete mpWindow;
        mpWindow = NULL;
    }

	SAFE_DELETE(mpTimer);
    DestroyDXContext();
}

// initialize the CPUT system
//-----------------------------------------------------------------------------
CPUTResult CPUT_DX11::CPUTInitialize(const cString pCPUTResourcePath)
{
    // set where CPUT will look for it's button images, fonts, etc
    return SetCPUTResourceDirectory(pCPUTResourcePath);
}


// Set where CPUT will look for it's button images, fonts, etc
//-----------------------------------------------------------------------------
CPUTResult CPUT_DX11::SetCPUTResourceDirectory(const cString ResourceDirectory)
{
    // check to see if the specified directory is valid
    CPUTResult result = CPUT_SUCCESS;

    // resolve the directory to a full path
    cString fullPath;
    CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
    result = pServices->ResolveAbsolutePathAndFilename(ResourceDirectory, &fullPath);
    if(CPUTFAILED(result))
    {
        return result;
    }

    // check existence of directory
    result = pServices->DoesDirectoryExist(fullPath);
    if(CPUTFAILED(result))
    {
        return result;
    }

    // set the resource directory (absolute path)
    mResourceDirectory = fullPath;

    // tell the gui system where to look for it's resources
    // todo: do we want to force a flush/reload of all resources (i.e. change control graphics)
    result = CPUTGuiControllerDX11::GetController()->SetResourceDirectory(ResourceDirectory);

    return result;
}

// Handle keyboard events
//-----------------------------------------------------------------------------
CPUTEventHandledCode CPUT_DX11::CPUTHandleKeyboardEvent(CPUTKey key)
{
    // dispatch event to GUI to handle GUI triggers (if any)
    CPUTEventHandledCode handleCode = CPUTGuiControllerDX11::GetController()->HandleKeyboardEvent(key);

    // dispatch event to users HandleMouseEvent() method
    HEAPCHECK;
    handleCode = HandleKeyboardEvent(key);
    HEAPCHECK;

    return handleCode;
}

// Handle mouse events
//-----------------------------------------------------------------------------
CPUTEventHandledCode CPUT_DX11::CPUTHandleMouseEvent(int x, int y, int wheel, CPUTMouseState state)
{
    // dispatch event to GUI to handle GUI triggers (if any)
    CPUTEventHandledCode handleCode = CPUTGuiControllerDX11::GetController()->HandleMouseEvent(x,y,wheel,state);

    // dispatch event to users HandleMouseEvent() method if it wasn't consumed by the GUI
    if(CPUT_EVENT_HANDLED != handleCode)
    {
        HEAPCHECK;
        handleCode = HandleMouseEvent(x,y,wheel,state);
        HEAPCHECK;
    }

    return handleCode;
}


// Call appropriate OS create window call
//-----------------------------------------------------------------------------
CPUTResult CPUT_DX11::MakeWindow(const cString WindowTitle, int windowWidth, int windowHeight, int windowX, int windowY)
{
    CPUTResult result;

    HEAPCHECK;

    // if we have a window, destroy it
    if(mpWindow)
    {
        delete mpWindow;
        mpWindow = NULL;
    }

    HEAPCHECK;

    // create the OS window
    mpWindow = new CPUTWindowWin();

    result = mpWindow->Create((CPUT*)this, WindowTitle, windowWidth, windowHeight, windowX, windowY);

    HEAPCHECK;

    return result;
}

// Return the current GUI controller
//-----------------------------------------------------------------------------
CPUTGuiControllerDX11* CPUT_DX11::CPUTGetGuiController()
{
    return CPUTGuiControllerDX11::GetController();
}


// Create a DX11 context
//-----------------------------------------------------------------------------
CPUTResult CPUT_DX11::CreateDXContext(CPUTWindowCreationParams params)
{

    HRESULT hr = S_OK;
    CPUTResult result = CPUT_SUCCESS;

    // window params
    RECT rc;
    HWND hWnd = mpWindow->GetHWnd();
    GetClientRect( hWnd, &rc );
    UINT width = rc.right - rc.left;
    UINT height = rc.bottom - rc.top;

    // set up DirectX creation parameters
    mdriverType = D3D_DRIVER_TYPE_NULL;
    mfeatureLevel = D3D_FEATURE_LEVEL_11_0;
    mpD3dDevice = NULL;
    mpContext = NULL;
    mpSwapChain = NULL;
    mSwapChainBufferCount = params.deviceParams.swapChainBufferCount;
    mpBackBufferRTV = NULL;
    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_DRIVER_TYPE driverTypes[] =
    {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_WARP,
        D3D_DRIVER_TYPE_REFERENCE,
    };
    UINT numDriverTypes = ARRAYSIZE( driverTypes );

    // SRV's (shader resource views) require Structured Buffer
    // usage (D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) which was 
    // introduced in shader model 5 (directx 11.0)
    //
    D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
    UINT numFeatureLevels = ARRAYSIZE( featureLevels );

    // swap chain information
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory( &sd, sizeof( sd ) );
    sd.BufferCount = mSwapChainBufferCount;
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;

    mSwapChainFormat = params.deviceParams.swapChainFormat;
    sd.BufferDesc.Format = params.deviceParams.swapChainFormat;
    sd.BufferDesc.RefreshRate.Numerator = params.deviceParams.refreshRate;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = params.deviceParams.swapChainUsage;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;  // Number of MSAA samples
    sd.SampleDesc.Quality = 0;
    sd.Windowed = !params.startFullscreen;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

    // set the vsync parameter
    mSyncInterval = (0 != params.deviceParams.refreshRate)? 1 : 0;

    // walk devices and create device and swap chain on best matching piece of hardware
    bool functionalityTestPassed = false;
    for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )
    {
        mdriverType = driverTypes[driverTypeIndex];
        hr = D3D11CreateDeviceAndSwapChain(
            NULL,
            mdriverType,
            NULL,
            createDeviceFlags,
            featureLevels,
            numFeatureLevels,
            D3D11_SDK_VERSION,
            &sd,
            &mpSwapChain,
            &mpD3dDevice,
            &mfeatureLevel,
            &mpContext
        );
        if( SUCCEEDED( hr ) )
        {
            functionalityTestPassed = TestContextForRequiredFeatures();
            if(true == functionalityTestPassed)
            {
                break;
            }
            else
            {
                // context was created, but failed to have required features
                // release and destroy this context and created resources
                SAFE_RELEASE(mpSwapChain);
                SAFE_RELEASE(mpContext);
                SAFE_RELEASE(mpD3dDevice);
            }
        }
    }
    ASSERT( (SUCCEEDED(hr) && (true==functionalityTestPassed)), _L("Failed creating device and swap chain.") );
    if(!SUCCEEDED(hr) || !functionalityTestPassed)
    {
        CPUTOSServices::GetOSServices()->OpenMessageBox(_L("Required DirectX hardware support not present"), _L("Your system does not support the DirectX feature levels required for this sample."));
        exit(1); // exit app directly
    }
   
    // If the WARP or Reference rasterizer is being used, the performance is probably terrible.
    // we throw up a dialog right after drawing the loading screen in CPUTCreateWindowAndContext
    // warning about that perf problem

    // call the DeviceCreated callback/backbuffer/etc creation
    result = CreateContext();

    CPUTRenderStateBlock *pBlock = new CPUTRenderStateBlockDX11();
    pBlock->CreateNativeResources();
    CPUTRenderStateBlock::SetDefaultRenderStateBlock( pBlock );

    // Create the per-frame constant buffer.
    D3D11_BUFFER_DESC bd = {0};
    bd.ByteWidth = sizeof(CPUTFrameConstantBuffer);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    ID3D11Buffer *pPerFrameConstantBuffer;
    hr = (CPUT_DX11::GetDevice())->CreateBuffer( &bd, NULL, &pPerFrameConstantBuffer );
    ASSERT( !FAILED( hr ), _L("Error creating constant buffer.") );
    CPUTSetDebugName( pPerFrameConstantBuffer, _L("Per-Frame Constant buffer") );
    cString name = _L("$cbPerFrameValues");
    mpPerFrameConstantBuffer = new CPUTBufferDX11( name, pPerFrameConstantBuffer );
    CPUTAssetLibrary::GetAssetLibrary()->AddConstantBuffer( name, mpPerFrameConstantBuffer );
    SAFE_RELEASE(pPerFrameConstantBuffer); // We're done with it.  The CPUTBuffer now owns it.

    return result;
}

// This function tests a created DirectX context for specific features required for
// the framework, and possibly sample.  If your sample has specific hw features
// you wish to check for at creation time, you can add them here and have them
// tested at startup time.  If no contexts support your desired features, then
// the system will revert to the DX reference rasterizer, or barring that, 
// pop up a dialog and exit.
//-----------------------------------------------------------------------------
bool CPUT_DX11::TestContextForRequiredFeatures()
{
    // D3D11_RESOURCE_MISC_BUFFER_STRUCTURED check
    // attempt to create a 
    // create the buffer for the shader resource view
    D3D11_BUFFER_DESC desc;
    ZeroMemory( &desc, sizeof(desc) );
    desc.Usage = D3D11_USAGE_DEFAULT;
    // set the stride for one 'element' block of verts
    UINT m_VertexStride      = 4*sizeof(float);                 // size in bytes of a single element - this test case we'll use 4 floats 
    desc.ByteWidth           = 1 * m_VertexStride;              // size in bytes of entire buffer - this test case uses just one element
    desc.BindFlags           = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags      = 0;
    desc.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = m_VertexStride;

    ID3D11Buffer *pVertexBufferForSRV=NULL;
    D3D11_SUBRESOURCE_DATA resourceData;
    float pData[4] ={ 0.0f, 0.0f, 0.0f, 0.0f };

    ZeroMemory( &resourceData, sizeof(resourceData) );
    resourceData.pSysMem = pData;
    HRESULT hr = mpD3dDevice->CreateBuffer( &desc, &resourceData, &pVertexBufferForSRV );
    SAFE_RELEASE(pVertexBufferForSRV);
    if(!SUCCEEDED(hr))
    {
        // failed the feature test
        return false;
    }

    // add other required features here

    return true;
}

// Return the active D3D device used to create the context
//-----------------------------------------------------------------------------
ID3D11Device* CPUT_DX11::GetDevice()
{
    return mpD3dDevice;
}

// Default creation routine for making the back/stencil buffers
//-----------------------------------------------------------------------------
CPUTResult CPUT_DX11::CreateContext()
{
    HRESULT hr;
    CPUTResult result;
    RECT rc;
    HWND hWnd = mpWindow->GetHWnd();

    GetClientRect( hWnd, &rc );
    UINT width = rc.right - rc.left;
    UINT height = rc.bottom - rc.top;

    // Create a render target view
    ID3D11Texture2D *pBackBuffer = NULL;
    hr = mpSwapChain->GetBuffer( 0, __uuidof( ID3D11Texture2D ), ( LPVOID* )&pBackBuffer );
    ASSERT( SUCCEEDED(hr), _L("Failed getting back buffer.") );

    hr = mpD3dDevice->CreateRenderTargetView( pBackBuffer, NULL, &mpBackBufferRTV );
    pBackBuffer->Release();
    ASSERT( SUCCEEDED(hr), _L("Failed creating render target view.") );
    CPUTSetDebugName( mpBackBufferRTV, _L("BackBufferView") );

    // create depth/stencil buffer
    result = CreateAndBindDepthBuffer(width, height);
    ASSERT( SUCCEEDED(hr), _L("Failed creating and binding depth buffer.") );

    // Setup the viewport
    D3D11_VIEWPORT vp;
    vp.Width = (FLOAT)width;
    vp.Height = (FLOAT)height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    mpContext->RSSetViewports( 1, &vp );

    return CPUT_SUCCESS;
}

// destroy the DX context and release all resources
//-----------------------------------------------------------------------------
CPUTResult CPUT_DX11::DestroyDXContext()
{
    if (mpContext) {
        mpContext->ClearState();
        mpContext->Flush();
    }

    SAFE_RELEASE( mpBackBufferRTV );
    SAFE_RELEASE( mpDepthStencilBuffer );
    SAFE_RELEASE( mpDepthStencilState );
    SAFE_RELEASE( mpDepthStencilView );
    SAFE_RELEASE( mpContext );
    SAFE_RELEASE( mpD3dDevice );
    SAFE_RELEASE( mpSwapChain );

    return CPUT_SUCCESS;
}

// Toggle the fullscreen mode
// This routine keeps the current desktop resolution.  DougB suggested allowing
// one to go fullscreen in a different resolution
//-----------------------------------------------------------------------------
CPUTResult CPUT_DX11::CPUTToggleFullScreenMode()
{    
    // get the current fullscreen state
    bool bIsFullscreen = CPUTGetFullscreenState();
     
    // toggle the state
    bIsFullscreen = !bIsFullscreen;

    // set the fullscreen state
    HRESULT hr = mpSwapChain->SetFullscreenState(bIsFullscreen, NULL);
    ASSERT( SUCCEEDED(hr), _L("Failed toggling full screen mode.") );

    // trigger resize event so that all buffers can resize
    int x,y,width,height;
    CPUTOSServices::GetOSServices()->GetClientDimensions(&x, &y, &width, &height);
    ResizeWindow(width,height);

    // trigger a fullscreen mode change call if the sample has decided to handle the mode change
    FullscreenModeChange( bIsFullscreen );

    return CPUT_SUCCESS;
}

// Set the fullscreen mode to a desired state
//-----------------------------------------------------------------------------
void CPUT_DX11::CPUTSetFullscreenState(bool bIsFullscreen)
{
    // get the current fullscreen state
    bool bCurrentFullscreenState = CPUTGetFullscreenState();
    if((bool)bCurrentFullscreenState == bIsFullscreen)
    {
        // no need to call expensive state change, full screen state is already
        // in desired state
        return;
    }

    // set the fullscreen state
    HRESULT hr = mpSwapChain->SetFullscreenState(bIsFullscreen, NULL);
    ASSERT( SUCCEEDED(hr), _L("Failed toggling full screen mode.") );

    // trigger resize event so that all buffers can resize
    int x,y,width,height;
    CPUTOSServices::GetOSServices()->GetClientDimensions(&x, &y, &width, &height);
    ResizeWindow(width,height);

    // trigger a fullscreen mode change call if the sample has decided to handle the mode change
    FullscreenModeChange( bIsFullscreen );
}

// Get a bool indicating whether the system is in full screen mode or not
//-----------------------------------------------------------------------------
bool CPUT_DX11::CPUTGetFullscreenState()
{
    // get the current fullscreen state
    BOOL bCurrentlyFullscreen;
    IDXGIOutput *pSwapTarget=NULL;
    mpSwapChain->GetFullscreenState(&bCurrentlyFullscreen, &pSwapTarget);
    SAFE_RELEASE(pSwapTarget);
    if(TRUE == bCurrentlyFullscreen )
    {
        return true;
    }
    return false;
}

// Create the depth buffer
//-----------------------------------------------------------------------------
CPUTResult CPUT_DX11::CreateAndBindDepthBuffer(int width, int height)
{
    HRESULT hr;

    // Clamp to minimum size of 1x1 pixel
    width  = max( width, 1 );
    height = max( height, 1 );

    // ---- DEPTH BUFFER ---
    // 1. Initialize the description of the depth buffer.
    D3D11_TEXTURE2D_DESC depthBufferDesc;
    ZeroMemory(&depthBufferDesc, sizeof(depthBufferDesc));

    // Set up the description of the depth buffer.
    depthBufferDesc.Width = width;
    depthBufferDesc.Height = height;
    depthBufferDesc.MipLevels = 1;
    depthBufferDesc.ArraySize = 1;
    depthBufferDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    depthBufferDesc.SampleDesc.Count = 1;
    depthBufferDesc.SampleDesc.Quality = 0;
    depthBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    depthBufferDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    depthBufferDesc.CPUAccessFlags = 0;
    depthBufferDesc.MiscFlags = 0;

    // Create the texture for the depth buffer using the filled out description.
    hr = mpD3dDevice->CreateTexture2D(&depthBufferDesc, NULL, &mpDepthStencilBuffer);
    ASSERT( SUCCEEDED(hr), _L("Failed to create texture.") );
    CPUTSetDebugName( mpDepthStencilBuffer, _L("DepthBufferTexture") );

    // 2. Initialize the description of the stencil state.
    D3D11_DEPTH_STENCIL_DESC depthStencilDesc;	
    ZeroMemory(&depthStencilDesc, sizeof(depthStencilDesc));

    // Set up the description of the stencil state.
    depthStencilDesc.DepthEnable = true;
    depthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    depthStencilDesc.DepthFunc = D3D11_COMPARISON_GREATER_EQUAL;

    depthStencilDesc.StencilEnable = true;
    depthStencilDesc.StencilReadMask = 0xFF;
    depthStencilDesc.StencilWriteMask = 0xFF;

    // Stencil operations if pixel is front-facing.
    depthStencilDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    depthStencilDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_INCR;
    depthStencilDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    depthStencilDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;

    // Stencil operations if pixel is back-facing.
    depthStencilDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    depthStencilDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_DECR;
    depthStencilDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    depthStencilDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;

    // Create the depth stencil state.
    hr = mpD3dDevice->CreateDepthStencilState(&depthStencilDesc, &mpDepthStencilState);
    ASSERT( SUCCEEDED(hr), _L("Failed to create depth-stencil state.") );
    mpContext->OMSetDepthStencilState(mpDepthStencilState, 1);

    // Create shader resource view
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
    hr = mpD3dDevice->CreateShaderResourceView(mpDepthStencilBuffer, &srvDesc, &mpDepthStencilSRV);
    ASSERT( SUCCEEDED(hr), _L("Failed to create depth-stencil SRV.") );
    CPUTSetDebugName( mpDepthStencilSRV, _L("DepthStencilSRV") );

    // Create the depth stencil view.
    D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
    ZeroMemory(&depthStencilViewDesc, sizeof(depthStencilViewDesc));
    depthStencilViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    depthStencilViewDesc.Texture2D.MipSlice = 0;
    hr = mpD3dDevice->CreateDepthStencilView(mpDepthStencilBuffer, &depthStencilViewDesc, &mpDepthStencilView);
    ASSERT( SUCCEEDED(hr), _L("Failed to create depth-stencil view.") );
    CPUTSetDebugName( mpDepthStencilView, _L("DepthStencilView") );

    // Bind the render target view and depth stencil buffer to the output render pipeline.
    mpContext->OMSetRenderTargets(1, &mpBackBufferRTV, mpDepthStencilView);

    CPUTRenderTargetColor::SetActiveRenderTargetView( mpBackBufferRTV );
    CPUTRenderTargetDepth::SetActiveDepthStencilView( mpDepthStencilView );

    return CPUT_SUCCESS;
}

// incoming resize event to be handled and translated
//-----------------------------------------------------------------------------
void CPUT_DX11::ResizeWindow(UINT width, UINT height)
{
    HRESULT hr;
    CPUTResult result;
    CPUTAssetLibraryDX11 *pAssetLibrary = (CPUTAssetLibraryDX11*)CPUTAssetLibraryDX11::GetAssetLibrary();

    // TODO: Making the back and depth buffers into CPUTRenderTargets should simplify this (CPUTRenderTarget* manages RTV, SRV, UAV, etc.)
    if( mpBackBuffer )         ((CPUTBufferDX11*)mpBackBuffer)->ReleaseBuffer();
    if( mpDepthBuffer )        ((CPUTBufferDX11*)mpDepthBuffer)->ReleaseBuffer();
    if( mpBackBufferTexture )  ((CPUTTextureDX11*)mpBackBufferTexture)->ReleaseTexture();
    if( mpDepthBufferTexture ) ((CPUTTextureDX11*)mpDepthBufferTexture)->ReleaseTexture();

    // Make sure we don't have any buffers bound.
    mpContext->ClearState();
    Present();
    mpContext->Flush();

    SAFE_RELEASE(mpBackBufferRTV);
    SAFE_RELEASE(mpBackBufferSRV);
    SAFE_RELEASE(mpBackBufferUAV);
    SAFE_RELEASE(mpDepthStencilSRV);

    CPUT::ResizeWindow( width, height );

    // Call the sample's clean up code if present.
    ReleaseSwapChain();

    // handle the internals of a resize
    int windowWidth, windowHeight;
    CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
    pServices->GetClientDimensions( &windowWidth, &windowHeight);

    // resize the swap chain
    hr = mpSwapChain->ResizeBuffers(mSwapChainBufferCount, windowWidth, windowHeight, mSwapChainFormat, 0);
    ASSERT( SUCCEEDED(hr), _L("Error resizing swap chain") );

    // re-create the render-target view
    ID3D11Texture2D *pSwapChainBuffer = NULL;
    hr = mpSwapChain->GetBuffer( 0, __uuidof( ID3D11Texture2D), (LPVOID*) (&pSwapChainBuffer));
    ASSERT(SUCCEEDED(hr), _L(""));
    hr = mpD3dDevice->CreateRenderTargetView( pSwapChainBuffer, NULL, &mpBackBufferRTV);
    ASSERT(SUCCEEDED(hr), _L(""));
    hr = mpD3dDevice->CreateShaderResourceView( pSwapChainBuffer, NULL, &mpBackBufferSRV);
    ASSERT(SUCCEEDED(hr), _L(""));
#ifdef CREATE_SWAP_CHAIN_UAV
	// Not every DXGI format supports UAV.  So, create UAV only if sample chooses to do so.
    hr = mpD3dDevice->CreateUnorderedAccessView( pSwapChainBuffer, NULL, &mpBackBufferUAV);
    ASSERT(SUCCEEDED(hr), _L(""));
#endif
    // Add the back buffer to the asset library.  Create CPUTBuffer and a CPUTTexture forms and add them.
    if( mpBackBuffer )
    {
        ((CPUTBufferDX11*)mpBackBuffer)->SetBufferAndViews( NULL, mpBackBufferSRV, mpBackBufferUAV );
    }
    else
    {
        cString backBufferName = _L("$BackBuffer"); 
        mpBackBuffer  = new CPUTBufferDX11( backBufferName,  NULL, mpBackBufferUAV );
        pAssetLibrary->AddBuffer( backBufferName, mpBackBuffer );
    }
    if( mpBackBufferTexture )
    {
        ((CPUTTextureDX11*)mpBackBufferTexture)->SetTextureAndShaderResourceView( NULL, mpBackBufferSRV );
    }
    else
    {
        cString backBufferName = _L("$BackBuffer"); 
        mpBackBufferTexture  = new CPUTTextureDX11( backBufferName, NULL, mpBackBufferSRV );
        pAssetLibrary->AddTexture( backBufferName, mpBackBufferTexture );
    }

    // release the old depth buffer objects
    // release the temporary swap chain buffer
    SAFE_RELEASE(pSwapChainBuffer);
    SAFE_RELEASE(mpDepthStencilBuffer);
    SAFE_RELEASE(mpDepthStencilState);
    SAFE_RELEASE(mpDepthStencilView);

    result = CreateAndBindDepthBuffer(windowWidth, windowHeight);
    if(CPUTFAILED(result))
    {
        // depth buffer creation error
        ASSERT(0,_L(""));
    }

    if( mpDepthBuffer )
    {
        ((CPUTBufferDX11*)mpDepthBuffer)->SetBufferAndViews( NULL, mpDepthStencilSRV, NULL );
    }
    else
    {
        cString depthBufferName = _L("$DepthBuffer"); 
        mpDepthBuffer  = new CPUTBufferDX11( depthBufferName, NULL, mpDepthStencilSRV );
        pAssetLibrary->AddBuffer( depthBufferName, mpDepthBuffer );
    }
    if( mpDepthBufferTexture )
    {
        ((CPUTTextureDX11*)mpDepthBufferTexture)->SetTextureAndShaderResourceView( NULL, mpDepthStencilSRV );
    }
    else
    {
        cString DepthBufferName = _L("$DepthBuffer"); 
        mpDepthBufferTexture  = new CPUTTextureDX11( DepthBufferName, NULL, mpDepthStencilSRV );
        pAssetLibrary->AddTexture( DepthBufferName, mpDepthBufferTexture );
    }

    // Release our extra reference to each view.
    // if(mpBackBufferSRV)   mpBackBufferSRV->Release();
    // if(mpBackBufferUAV)   mpBackBufferUAV->Release();
    // if(mpDepthStencilSRV) mpDepthStencilSRV->Release();;

    // set the viewport
    D3D11_VIEWPORT vp;
    vp.Width = (FLOAT) windowWidth;
    vp.Height = (FLOAT)windowHeight;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    mpContext->RSSetViewports( 1, &vp );

    // trigger the GUI manager to resize
    CPUTGuiControllerDX11::GetController()->Resize();
}

// 'soft' resize - just stretch-blit
//-----------------------------------------------------------------------------
void CPUT_DX11::ResizeWindowSoft(UINT width, UINT height)
{
    UNREFERENCED_PARAMETER(width);
    UNREFERENCED_PARAMETER(height);
    // trigger the GUI manager to resize
    CPUTGuiControllerDX11::GetController()->Resize();

    InnerExecutionLoop();
}

//-----------------------------------------------------------------------------
void CPUT_DX11::SetPerFrameConstantBuffer( double totalSeconds )
{
    if( mpPerFrameConstantBuffer )
    {
        ID3D11Buffer *pBuffer = mpPerFrameConstantBuffer->GetNativeBuffer();

        // update parameters of constant buffer
        D3D11_MAPPED_SUBRESOURCE mapInfo;
        mpContext->Map( pBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapInfo );
        {
            // TODO: remove construction of XMM type
            CPUTFrameConstantBuffer *pCb = (CPUTFrameConstantBuffer*)mapInfo.pData;
            CPUTCamera *pCamera     = gpSample->GetCamera();
            if( pCamera )
            {
                pCb->View               = XMMATRIX((float*)pCamera->GetViewMatrix());
                pCb->Projection         = XMMATRIX((float*)pCamera->GetProjectionMatrix());
            }
            pCb->LightColor         = XMLoadFloat3(&XMFLOAT3((float*)&mLightColor)); // TODO: Get from light
            float totalSecondsFloat = (float)totalSeconds;
            pCb->TotalSeconds       = XMLoadFloat(&totalSecondsFloat);
            pCb->AmbientColor       = XMLoadFloat3(&XMFLOAT3((float*)&mAmbientColor));
        }
        mpContext->Unmap(pBuffer,0);
    }
}

// Call the user's Render() callback (if it exists)
//-----------------------------------------------------------------------------
void CPUT_DX11::InnerExecutionLoop()
{
#ifdef CPUT_GPA_INSTRUMENTATION
    D3DPERF_BeginEvent(D3DCOLOR(0xff0000), L"CPUT User's Render() ");
#endif
    if(!mbShutdown)
    {
		if( mpSwapChain )
		{
			double deltaSeconds = mpTimer->GetElapsedTime();
			Update(deltaSeconds);
			Present(); // Note: Presenting immediately before Rendering minimizes CPU stalls (i.e., execute Update() before Present() stalls)

			double totalSeconds = mpTimer->GetTotalTime();
			SetPerFrameConstantBuffer(totalSeconds);
			CPUTMaterialDX11::ResetStateTracking();
			Render(deltaSeconds);
		}
        if(!CPUTOSServices::GetOSServices()->DoesWindowHaveFocus())
        {
            Sleep(100);
        }
    }
    else
    {
#ifndef _DEBUG
        exit(0);
#endif
        Present(); // Need to present, or will leak all references held by previous Render()!
        ShutdownAndDestroy();
    }

#ifdef CPUT_GPA_INSTRUMENTATION
    D3DPERF_EndEvent();
#endif
}

// draw all the GUI controls
//-----------------------------------------------------------------------------
void CPUT_DX11::CPUTDrawGUI()
{
#ifdef CPUT_GPA_INSTRUMENTATION
    D3DPERF_BeginEvent(D3DCOLOR(0xff0000), L"CPUT Draw GUI");
#endif

    // draw all the Gui controls
    HEAPCHECK;
        CPUTGuiControllerDX11::GetController()->Draw(mpContext, mSyncInterval);
    HEAPCHECK;

#ifdef CPUT_GPA_INSTRUMENTATION
        D3DPERF_EndEvent();
#endif
}

// Parse the command line for the parameters
// Only parameters that are specified on the command line are updated, if there
// are no parameters for a value, the previous WindowParams settings passed in
// are preserved
//-----------------------------------------------------------------------------
CPUTResult CPUT_DX11::CPUTParseCommandLine(cString commandLine, CPUTWindowCreationParams *pWindowParams, cString *pFilename)
{
    ASSERT( (NULL!=pWindowParams), _L("Required command line parsing parameter is NULL"));
    ASSERT( (NULL!=pFilename), _L("Required command line parsing parameter is NULL"));
   
    // there are no command line parameters, just return
    if(0==commandLine.size())
    {
        return CPUT_SUCCESS;
    }

    // we do have parameters, so parse them.
#if defined (UNICODE) || defined(_UNICODE)
    // convert command line to lowercase version
    cString CommandLineParams(commandLine);        
    std::transform(CommandLineParams.begin(), CommandLineParams.end(), CommandLineParams.begin(), ::tolower);

    // special case - double-clicked on a file
    // In the case someone has associated .set files with CPUT, then the target set file comes in surrounded
    // by quote marks (i.e. "c:\mySample\Asset\City.set").  If the first char is a quote mark, we're in this
    // special case
    if('"' == CommandLineParams[0])
    {
        pFilename->assign(&CommandLineParams[1]);   // remove the leading " char
        (*pFilename)[pFilename->size()-1] = '\0';   // remove the trailing " char
        return CPUT_SUCCESS;
    }

    wchar_t separators[]   = L" \t\n";
    wchar_t* nextToken = NULL;
    wchar_t* token = wcstok_s((wchar_t*)CommandLineParams.c_str(), separators, &nextToken);
    while(token) 
    {
        if('-' == token[0])
        {
            // parameter - get next token for which one
            cString ParameterName(&token[1]);
            if(0!=ParameterName.size())
            {
                if(0==ParameterName.compare(_L("width")))
                {
                    // get the next value
                    token = wcstok_s(NULL, separators, &nextToken); 
                    ASSERT(token, _L("-width command line parameter missing required numerical value"));
                    pWindowParams->windowWidth = _wtoi(token);
                }
                else if(0==ParameterName.compare(_L("height")))
                {
                    // get the next value
                    token = wcstok_s(NULL, separators, &nextToken); 
                    ASSERT(token, _L("-height command line parameter missing required numerical value"));
                    pWindowParams->windowHeight = _wtoi(token);
                }
                else if(0==ParameterName.compare(_L("fullscreen")))
                {
                    // get the bool 
                    token = wcstok_s(NULL, separators, &nextToken); 
                    cString boolString(token);
                    if(0==boolString.compare(_L("true")))
                    {
                        pWindowParams->startFullscreen = true;
                    }
                }
                else if(0==ParameterName.compare(_L("vsync")))
                {
                    // get the bool 
                    token = wcstok_s(NULL, separators, &nextToken); 
                    cString boolString(token);
                    if( (0==boolString.compare(_L("on"))) || (0==boolString.compare(_L("true"))) )
                    {
                        // vsync set to 30 FPS
                        pWindowParams->deviceParams.refreshRate = 30;
                    }
					if( (0==boolString.compare(_L("off"))) || (0==boolString.compare(_L("false"))) )
					{
						pWindowParams->deviceParams.refreshRate = 0;
					}
                }
                else if(0==ParameterName.compare(_L("xpos")))
                {
                    // get the next value
                    token = wcstok_s(NULL, separators, &nextToken); 
                    ASSERT(token, _L("-xpos command line parameter missing required numerical value"));
                    pWindowParams->windowPositionX = _wtoi(token);
                }
                else if(0==ParameterName.compare(_L("ypos")))
                {
                    // get the next value
                    token = wcstok_s(NULL, separators, &nextToken); 
                    ASSERT(token, _L("-ypos command line parameter missing required numerical value"));
                    pWindowParams->windowPositionY = _wtoi(token);
                }
                else if(0==ParameterName.compare(_L("file")))
                {
                    // get the filename 
                    token = wcstok_s(NULL, separators, &nextToken);
                    pFilename->assign(token);
                }
                else
                {
                    // we don't know what the string was, but all specified ones should be of the form
                    // '-<keyword> <value>' 
                    // so skip over the <value> part
                    token = wcstok_s(NULL, separators, &nextToken); 
                }
                
                // we don't know what this parameter is, let user parse it
            }
        }
        
        // advance to next token
        token = wcstok_s(NULL, separators, &nextToken);
    }
#else
    ASSERT(false, _L("CPUT_DX11::CPUTParseCommandLine non-UNICODE version not written yet - need to write if we want to support multi-byte"));
#endif     
    
    return CPUT_SUCCESS;
}

// Create a window context
//-----------------------------------------------------------------------------
CPUTResult CPUT_DX11::CPUTCreateWindowAndContext(const cString WindowTitle, CPUTWindowCreationParams windowParams)
{
    CPUTResult result = CPUT_SUCCESS;

    HEAPCHECK;

	// create the window
    result = MakeWindow(WindowTitle, windowParams.windowWidth, windowParams.windowHeight, windowParams.windowPositionX, windowParams.windowPositionY);
    if(CPUTFAILED(result))
    {
        return result;
    }

    HEAPCHECK;

    // create the DX context
    result = CreateDXContext(windowParams);
    if(CPUTFAILED(result))
    {
        return result;
    }

    HEAPCHECK;
#define ENABLE_GUI
#ifdef ENABLE_GUI
    // initialize the gui controller 
    // Use the ResourceDirectory that was given during the Initialize() function
    // to locate the GUI+font resources
    CPUTGuiControllerDX11 *pGUIController = CPUTGuiControllerDX11::GetController();
    cString ResourceDirectory = GetCPUTResourceDirectory();
    result = pGUIController->Initialize(mpContext, ResourceDirectory);
    if(CPUTFAILED(result))
    {
        return result;
    }
    // register the callback object for GUI events as our sample
    CPUTGuiControllerDX11::GetController()->SetCallback(this);
#endif
    HEAPCHECK;
    DrawLoadingFrame();
    HEAPCHECK;
    
    // warn the user they are using the software rasterizer
    if((D3D_DRIVER_TYPE_REFERENCE == mdriverType) || (D3D_DRIVER_TYPE_WARP == mdriverType))
    {
        CPUTOSServices::GetOSServices()->OpenMessageBox(_L("Performance warning"), _L("Your graphics hardware does not support the DirectX features required by this sample. The sample is now running using the DirectX software rasterizer."));
    }


    // trigger a post-create user callback event
    HEAPCHECK;
    Create();
    HEAPCHECK;

	//
	// Start the timer after everything is initialized and assets have been loaded
	//
	mpTimer->StartTimer();

    // if someone triggers the shutdown routine in on-create, exit
    if(mbShutdown)
    {
        return result;
    }

    // fill first frame with clear values so render order later is ok
    const float srgbClearColor[] = { 0.0993f, 0.0993f, 0.0993f, 1.0f };
    mpContext->ClearRenderTargetView( mpBackBufferRTV, srgbClearColor );
    mpContext->ClearDepthStencilView(mpDepthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 0.0f, 0);

    // trigger a 'resize' event
    int x,y,width,height;
    CPUTOSServices::GetOSServices()->GetClientDimensions(&x, &y, &width, &height);
    ResizeWindow(width,height);

    return result;
}

// Pop up a message box with specified title/text
//-----------------------------------------------------------------------------
void CPUT_DX11::DrawLoadingFrame()
{
    // fill first frame with clear values so render order later is ok
    const float srgbClearColor[] = { 0.0993f, 0.0993f, 0.0993f, 1.0f };
    mpContext->ClearRenderTargetView( mpBackBufferRTV, srgbClearColor );
    mpContext->ClearDepthStencilView(mpDepthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 0.0f, 0);

    // get center
    int x,y,width,height;
    CPUTOSServices::GetOSServices()->GetClientDimensions(&x, &y, &width, &height);

    // draw "loading..." text
    CPUTGuiControllerDX11 *pGUIController = CPUTGuiControllerDX11::GetController();
    CPUTText *pText = NULL;
    pGUIController->CreateText(_L("Just a moment, now loading..."), 999, 0, &pText);
    pGUIController->EnableAutoLayout(false);
    int textWidth, textHeight;
    pText->GetDimensions(textWidth, textHeight);
    pText->SetPosition(width/2-textWidth/2, height/2);

    pGUIController->Draw(mpContext);
    pGUIController->DeleteAllControls();
    pGUIController->EnableAutoLayout(true);
    
    // present loading screen
    mpSwapChain->Present( mSyncInterval, 0 );
}

// Pop up a message box with specified title/text
//-----------------------------------------------------------------------------
void CPUT_DX11::CPUTMessageBox(const cString DialogBoxTitle, const cString DialogMessage)
{
    CPUTOSServices::GetOSServices()->OpenMessageBox(DialogBoxTitle.c_str(), DialogMessage.c_str());
}

// start main message loop
//-----------------------------------------------------------------------------
int CPUT_DX11::CPUTMessageLoop()
{
#ifdef CPUT_GPA_INSTRUMENTATION
    D3DPERF_BeginEvent(D3DCOLOR(0xff0000), L"CPUTMessageLoop");
#endif

    return mpWindow->StartMessageLoop();

#ifdef CPUT_GPA_INSTRUMENTATION
    D3DPERF_EndEvent();
#endif
}

// Window is closing. Shut the system to shut down now, not later.
//-----------------------------------------------------------------------------
void CPUT_DX11::DeviceShutdown()
{
    if(mpSwapChain)
    {
        // DX requires setting fullscreenstate to false before exit.
        mpSwapChain->SetFullscreenState(false, NULL);
    }
    if(false == mbShutdown)
    {
        mbShutdown = true;
        ShutdownAndDestroy();
    }
}

// Shutdown the CPUT system
// Destroy all 'global' resource handling objects, all asset handlers,
// the DX context, and everything EXCEPT the window
//-----------------------------------------------------------------------------
void CPUT_DX11::Shutdown()
{
    // release the lock on the mouse (if there was one)
    CPUTOSServices::GetOSServices()->ReleaseMouse();
    mbShutdown = true;
}

// Frees all resources and removes all assets from asset library
//-----------------------------------------------------------------------------
void CPUT_DX11::RestartCPUT()
{
    //
    // Clear out all CPUT resources
    //
    CPUTInputLayoutCacheDX11::GetInputLayoutCache()->ClearLayoutCache();
    CPUTAssetLibrary::GetAssetLibrary()->ReleaseAllLibraryLists();
	CPUTGuiControllerDX11::GetController()->DeleteAllControls();
	CPUTGuiControllerDX11::GetController()->ReleaseResources();

    //
    // Clear out all DX resources and contexts
    //
    DestroyDXContext();

    //
    // Signal the window to close
    //
    mpWindow->Destroy();
	
	//
	// Clear out the timer
	//
	mpTimer->StopTimer();
	mpTimer->ResetTimer();
    
    HEAPCHECK;
}
// Actually destroy all 'global' resource handling objects, all asset handlers,
// the DX context, and everything EXCEPT the window
//-----------------------------------------------------------------------------
void CPUT_DX11::ShutdownAndDestroy()
{
    // make sure no more rendering can happen
    mbShutdown = true;

    // call the user's OnShutdown code
    Shutdown();
    CPUTInputLayoutCacheDX11::DeleteInputLayoutCache();
    CPUTAssetLibraryDX11::DeleteAssetLibrary();
    CPUTGuiControllerDX11::DeleteController();

// #ifdef _DEBUG
#if 0
    ID3D11Debug *pDebug;
    mpD3dDevice->QueryInterface(IID_ID3D11Debug, (VOID**)(&pDebug));
    if( pDebug )
    {
        pDebug->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL);
        pDebug->Release();
    }
#endif
    CPUTOSServices::DeleteOSServices();

    // Tell the window layer to post a close-window message to OS
    // and stop the message pump
    mpWindow->Destroy();

    HEAPCHECK;
}

//-----------------------------------------------------------------------------
void CPUTSetDebugName( void *pResource, cString name )
{
#ifdef _DEBUG
    char pCharString[CPUT_MAX_STRING_LENGTH];
    const wchar_t *pWideString = name.c_str();
    UINT ii;
    UINT length = min( (UINT)name.length(), (CPUT_MAX_STRING_LENGTH-1));
    for(ii=0; ii<length; ii++)
    {
        pCharString[ii] = (char)pWideString[ii];
    }
    pCharString[ii] = 0; // Force NULL termination
    ((ID3D11DeviceChild*)pResource)->SetPrivateData( WKPDID_D3DDebugObjectName, (UINT)name.length(), pCharString );
#endif // _DEBUG
}
