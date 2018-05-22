//-------------------------------------------------------------------------------------------------
// File: Engine.cpp
// Author: Ben Mistal
// Copyright 2010-2012 Mistal Research, Inc.
//-------------------------------------------------------------------------------------------------
#include "Engine.h"
#include "Noise.h"

CEngine::CEngine()
{
	m_bShowWireframe					= false;
	m_bFullScreen						= false;
	m_bHardwareDevice					= false;

	m_hWnd								= NULL;
	m_driverType						= D3D_DRIVER_TYPE_NULL;
	m_featureLevel						= D3D_FEATURE_LEVEL_11_0;
	m_pd3dDevice						= NULL;
	m_pImmediateContext					= NULL;
	m_pSwapChain						= NULL;
	m_pBackBuffer						= NULL;
	m_pRenderTargetView					= NULL;
	m_pDepthStencilTexture				= NULL;
	m_pDepthStencilTextureDSV			= NULL;

	m_pRegionVertexLayout				= NULL;
	m_pRegionPassThroughVertexShader	= NULL;

	m_pRegionSplitGeometryShader		= NULL;
	m_pRegionFaceGeometryShader			= NULL;
	m_pRegionWireGeometryShader			= NULL;
	m_pRegionPixelShader				= NULL;
	m_pRegionWireframePixelShader		= NULL;

	m_pPositionTexture					= NULL;
	m_pPositionTextureRV				= NULL;
	m_pPositionRenderTargetView			= NULL;

	m_pCompositeVertexShader			= NULL;
	m_pBackgroundPixelShader			= NULL;
	m_pCompositePixelShader				= NULL;

	m_pCompositeVertexLayout			= NULL;
	m_pCompositeVertexBuffer			= NULL;
	m_pCompositeIndexBuffer				= NULL;	

	for( unsigned int nIndex= 0; nIndex < NumTempVertexBuffers; nIndex++ )
	{
		m_pRegionVertexBufferArray[ nIndex ] = NULL;

	} // end for( unsigned int nIndex= 0; nIndex < NumTempVertexBuffers; nIndex++ )

	m_pRasterizerState				= NULL;
	m_pRegionDepthStencilState		= NULL;
	m_pBackgroundDepthStencilState	= NULL;
	m_pCompositeDepthStencilState	= NULL;

	m_pConstantBuffer				= NULL;
	m_pControlConstantBuffer		= NULL;

	m_p2DValueGradientTexture			= NULL;
	m_p2DValueGradientTextureResource	= NULL;
	m_p2DValueGradientTextureRV			= NULL;

	m_pSamplerLinearWrap			= NULL;
	m_pSamplerPointClamp			= NULL;
	m_pSamplerPointWrap				= NULL;

	memset( &m_orientationInformation, 0, sizeof( OrientationInformation ) );

	memset( &m_constantBufferData, 0, sizeof( ConstantBuffer ) );

	memset( &m_timerInformation, 0, sizeof( TimerInformation ) );
	QueryPerformanceFrequency( &m_timerInformation.m_nTimerFrequency );
	::QueryPerformanceCounter( &m_timerInformation.m_nLastPrintedStatTime );

} // end CEngine::CEngine( HWND hWnd )

CEngine::~CEngine()
{
	CleanupDevice();

} // end CEngine::~CEngine()

void CEngine::CleanupDevice()
{
    if( m_pImmediateContext ) m_pImmediateContext->ClearState();

	SAFE_RELEASE( m_p2DValueGradientTexture );
	SAFE_RELEASE( m_p2DValueGradientTextureResource );
	SAFE_RELEASE( m_p2DValueGradientTextureRV );

	SAFE_RELEASE( m_pSamplerLinearWrap );
	SAFE_RELEASE( m_pSamplerPointClamp );
	SAFE_RELEASE( m_pSamplerPointWrap );

	SAFE_RELEASE( m_pConstantBuffer );
	SAFE_RELEASE( m_pControlConstantBuffer );
	SAFE_RELEASE( m_pRasterizerState );
	SAFE_RELEASE( m_pRegionDepthStencilState );
	SAFE_RELEASE( m_pBackgroundDepthStencilState );
	SAFE_RELEASE( m_pCompositeDepthStencilState );

	for( unsigned int nIndex= 0; nIndex < NumTempVertexBuffers; nIndex++ )
	{
		SAFE_RELEASE( m_pRegionVertexBufferArray[ nIndex ] );

	} // end for( unsigned int nIndex= 0; nIndex < NumTempVertexBuffers; nIndex++ )

	SAFE_RELEASE( m_pRegionVertexLayout );
    SAFE_RELEASE( m_pRegionPassThroughVertexShader );

	SAFE_RELEASE( m_pRegionSplitGeometryShader );
	SAFE_RELEASE( m_pRegionFaceGeometryShader );
	SAFE_RELEASE( m_pRegionWireGeometryShader );
	SAFE_RELEASE( m_pRegionPixelShader );
	SAFE_RELEASE( m_pRegionWireframePixelShader );

	SAFE_RELEASE( m_pPositionTexture );
	SAFE_RELEASE( m_pPositionTextureRV );
	SAFE_RELEASE( m_pPositionRenderTargetView );

	SAFE_RELEASE( m_pCompositeVertexLayout );
	SAFE_RELEASE( m_pCompositeVertexBuffer );
	SAFE_RELEASE( m_pCompositeIndexBuffer );

    SAFE_RELEASE( m_pCompositeVertexShader );
	SAFE_RELEASE( m_pBackgroundPixelShader );
	SAFE_RELEASE( m_pCompositePixelShader );

    SAFE_RELEASE( m_pDepthStencilTexture );
    SAFE_RELEASE( m_pDepthStencilTextureDSV );
    SAFE_RELEASE( m_pRenderTargetView );
	SAFE_RELEASE( m_pBackBuffer );
    SAFE_RELEASE( m_pSwapChain );
    SAFE_RELEASE( m_pImmediateContext );
    SAFE_RELEASE( m_pd3dDevice );

	double fTimerFrequency = ( double )m_timerInformation.m_nTimerFrequency.QuadPart;

	LARGE_INTEGER nMinimumTime;
	LARGE_INTEGER nMaximumTime;
	LARGE_INTEGER nTotalTime;

	unsigned int nStoredTimerFrameCount = 0;
	nMinimumTime.QuadPart = 0;
	nMaximumTime.QuadPart = 0;
	nTotalTime.QuadPart = 0;

	CalculateTimerStats( nStoredTimerFrameCount, nMinimumTime, nMaximumTime, nTotalTime );

	if( m_timerInformation.m_nTimerFrequency.QuadPart > 0 )
	{
		char strOutput[ 1024 ];

		if( nMinimumTime.QuadPart > 0 )
		{
			sprintf_s(	strOutput,
						1024,
						"Fastest Frame Render Time = %fms (%f FPS)\n",
						( ( ( double )nMinimumTime.QuadPart / fTimerFrequency ) * 1000.0 ),
						( fTimerFrequency / ( double )nMinimumTime.QuadPart ) ); 

			::OutputDebugStringA( strOutput );

		} // end if( nMinimumTime.QuadPart > 0 )

		if( nMaximumTime.QuadPart > 0 )
		{
			sprintf_s(	strOutput,
						1024,
						"Slowest Frame Render Time = %fms (%f FPS)\n",
						( ( ( double )nMaximumTime.QuadPart / fTimerFrequency ) * 1000.0 ),
						( fTimerFrequency / ( double )nMaximumTime.QuadPart ) ); 

			::OutputDebugStringA( strOutput );

		} // end if( nMaximumTime.QuadPart > 0 )

		if( ( nTotalTime.QuadPart > 0 ) && ( nStoredTimerFrameCount > 0 ) )
		{
			double fAverageTime = ( double )nTotalTime.QuadPart / ( double )nStoredTimerFrameCount;

			sprintf_s(	strOutput,
						1024,
						"Average Frame Render Time = %fms (%f FPS)\n",
						( ( fAverageTime / fTimerFrequency ) * 1000.0 ),
						( fTimerFrequency / fAverageTime ) ); 

			::OutputDebugStringA( strOutput );

		} // end if( ( nTotalTime.QuadPart > 0 ) && ( nStoredTimerFrameCount > 0 ) )

		sprintf_s(	strOutput,
					1024,
					"Number of Frames = %d\n",
					m_timerInformation.m_nFrameCount ); 

		::OutputDebugStringA( strOutput );

	} // if( m_timerInformation.m_nTimerFrequency.QuadPart > 0 ) 
	
} // end void CEngine::CleanupDevice()

//--------------------------------------------------------------------------------------
// Create Direct3D device and swap chain, and all other required DX resources
//--------------------------------------------------------------------------------------
HRESULT CEngine::InitDevice( HWND hWnd )
{
	m_hWnd = hWnd;

    HRESULT hr = S_OK;

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_DRIVER_TYPE driverTypes[] =
    {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_WARP,
        D3D_DRIVER_TYPE_REFERENCE,

    }; // end D3D_DRIVER_TYPE driverTypes[]

    UINT numDriverTypes = ARRAYSIZE( driverTypes );

    D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_0,
 
    }; // end D3D_FEATURE_LEVEL featureLevels[]

    UINT numFeatureLevels = ARRAYSIZE( featureLevels );

    DXGI_SWAP_CHAIN_DESC sd;
    memset( &sd, 0, sizeof( DXGI_SWAP_CHAIN_DESC ) );
    sd.BufferCount							= 1;
    sd.BufferDesc.Width						= nHSize;
    sd.BufferDesc.Height					= nVSize;
    sd.BufferDesc.Format					= DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator		= 60;
    sd.BufferDesc.RefreshRate.Denominator	= 1;
    sd.BufferUsage							= DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow							= m_hWnd;
    sd.SampleDesc.Count						= 1;
    sd.SampleDesc.Quality					= 0;
    sd.Windowed								= m_bFullScreen ? FALSE : TRUE;

    for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )
    {
        m_driverType = driverTypes[driverTypeIndex];

        hr = D3D11CreateDeviceAndSwapChain(	NULL, 
											m_driverType, 
											NULL, 
											createDeviceFlags, 
											featureLevels, 
											numFeatureLevels,
											D3D11_SDK_VERSION, 
											&sd, 
											&m_pSwapChain, 
											&m_pd3dDevice, 
											&m_featureLevel, 
											&m_pImmediateContext );

        if( SUCCEEDED( hr ) )
		{
			m_bHardwareDevice = ( driverTypeIndex == 0 ) ? true : false;
            break;

		} // end if( SUCCEEDED( hr ) )

    } //end for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )

    if( FAILED( hr ) )
	{
        return hr;

	} // end if( FAILED( hr ) )

	SetFullScreenMode( m_bFullScreen );

 	//-------------------
    // Setup the viewport
 	//-------------------
    D3D11_VIEWPORT vp;
    memset( &vp, 0, sizeof( D3D11_VIEWPORT ) );
    vp.Width	= ( FLOAT )nHSize;
    vp.Height	= ( FLOAT )nVSize;
    vp.MinDepth	= 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;

    m_pImmediateContext->RSSetViewports( 1, &vp );

	hr = InitTextures();
    if( FAILED( hr ) )
	{
        return hr;

	} // end if( FAILED( hr ) )

	hr = InitBuffers();
    if( FAILED( hr ) )
	{
        return hr;

	} // end if( FAILED( hr ) )

	hr = InitShaders();
    if( FAILED( hr ) )
	{
        return hr;

	} // end if( FAILED( hr ) )

	hr = InitStateObjects();
    if( FAILED( hr ) )
	{
        return hr;

	} // end if( FAILED( hr ) )

	if( !m_bHardwareDevice )
	{
		::MessageBoxA( m_hWnd, "Warning, the DirectX 11 rendering device is not running in hardware.\nPlease press the spacebar to render each frame.", "TerrainTessellation", MB_OK );

	} // end if( !m_bHardwareDevice )

	return S_OK;

} // end HRESULT CEngine::InitDevice()

HRESULT CEngine::ResizeDevice()
{
	HRESULT hr = S_OK;

	if( m_pd3dDevice == NULL ) return S_FALSE;
	if( m_pSwapChain == NULL ) return S_FALSE;

	RECT rc;
    ::GetClientRect( m_hWnd, &rc );
	int nWidth = rc.right - rc.left;
	int nHeight = rc.bottom - rc.top;

	//-------------------------------------------------------
	// Release the existing viewport size dependent resources
	//-------------------------------------------------------
	SAFE_RELEASE( m_pPositionRenderTargetView );
	SAFE_RELEASE( m_pPositionTextureRV );
	SAFE_RELEASE( m_pPositionTexture );
	SAFE_RELEASE( m_pDepthStencilTextureDSV );
	SAFE_RELEASE( m_pDepthStencilTexture );
	SAFE_RELEASE( m_pRenderTargetView );
	SAFE_RELEASE( m_pBackBuffer );

	//----------------------
	// Resize the swap chain
	//----------------------
	hr = m_pSwapChain->ResizeBuffers( 1, nWidth, nHeight, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH );
    if( FAILED( hr ) )
	{
        return hr;

	} // end if( FAILED( hr ) )

 	//-------------------
    // Setup the viewport
 	//-------------------
    D3D11_VIEWPORT vp;
    memset( &vp, 0, sizeof( D3D11_VIEWPORT ) );
    vp.Width	= ( FLOAT )nWidth;
    vp.Height	= ( FLOAT )nHeight;
    vp.MinDepth	= 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;

    m_pImmediateContext->RSSetViewports( 1, &vp );

	hr = InitTextures();
    if( FAILED( hr ) )
	{
        return hr;

	} // end if( FAILED( hr ) )

	return S_OK;

} // end HRESULT CEngine::ResizeDevice()

HRESULT CEngine::InitTextures()
{
	HRESULT hr = S_OK;

	if( m_pd3dDevice == NULL ) return S_FALSE;
	if( m_pSwapChain == NULL ) return S_FALSE;

	RECT rc;
    ::GetClientRect( m_hWnd, &rc );
	int nWidth = rc.right - rc.left;
	int nHeight = rc.bottom - rc.top;

	//-----------------------------------------
    // Create the backbuffer render target view
	//-----------------------------------------
	if( m_pBackBuffer == NULL )
	{
		hr = m_pSwapChain->GetBuffer( 0, __uuidof( ID3D11Texture2D ), ( LPVOID* )&m_pBackBuffer );
		if( FAILED( hr ) )
		{
			return hr;

		} // end  if( FAILED( hr ) )

	} // end if( m_pBackBuffer == NULL )

	if( m_pRenderTargetView == NULL )
	{
		hr = m_pd3dDevice->CreateRenderTargetView( m_pBackBuffer, NULL, &m_pRenderTargetView );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pRenderTargetView == NULL )

	//-----------------------------
    // Create depth stencil texture
	//-----------------------------
	if( m_pDepthStencilTexture == NULL )
	{
		D3D11_TEXTURE2D_DESC descDepth;
		memset( &descDepth, 0, sizeof( D3D11_TEXTURE2D_DESC ) );
		descDepth.Width					= nWidth;
		descDepth.Height				= nHeight;
		descDepth.ArraySize				= 1;
		descDepth.MipLevels				= 1;
		descDepth.Format				= DXGI_FORMAT_D24_UNORM_S8_UINT;
		descDepth.SampleDesc.Count		= 1;
		descDepth.SampleDesc.Quality	= 0;
		descDepth.Usage					= D3D11_USAGE_DEFAULT;
		descDepth.BindFlags				= D3D11_BIND_DEPTH_STENCIL;
		descDepth.CPUAccessFlags		= 0;
		descDepth.MiscFlags				= 0;

		hr = m_pd3dDevice->CreateTexture2D( &descDepth, NULL, &m_pDepthStencilTexture );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pDepthStencilTexture == NULL )

 	//------------------------------
	// Create the depth stencil view
	//------------------------------
	if( m_pDepthStencilTextureDSV == NULL )
	{
		D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
		memset( &descDSV, 0, sizeof( D3D11_DEPTH_STENCIL_VIEW_DESC ) );
		descDSV.Format				= DXGI_FORMAT_D24_UNORM_S8_UINT;
		descDSV.ViewDimension		= D3D11_DSV_DIMENSION_TEXTURE2D;
		descDSV.Texture2D.MipSlice	= 0;

		hr = m_pd3dDevice->CreateDepthStencilView( m_pDepthStencilTexture, &descDSV, &m_pDepthStencilTextureDSV );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pDepthStencilTextureDSV == NULL )

 	//----------------------------
	// Create the Position Texture
 	//----------------------------
	if( m_pPositionTexture == NULL )
	{
		D3D11_TEXTURE2D_DESC positionTD;
		memset( &positionTD, 0, sizeof( D3D11_TEXTURE2D_DESC ) );
		positionTD.Width				= nWidth;
		positionTD.Height				= nHeight;
		positionTD.ArraySize			= 1;
		positionTD.MipLevels			= 1;
		positionTD.Format				= DXGI_FORMAT_R32G32B32A32_FLOAT;
		positionTD.SampleDesc.Count		= 1;
		positionTD.SampleDesc.Quality	= 0;
		positionTD.Usage				= D3D11_USAGE_DEFAULT;
		positionTD.BindFlags			= D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

		hr = m_pd3dDevice->CreateTexture2D( &positionTD, NULL, &m_pPositionTexture );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pPositionTexture == NULL )

 	//------------------------------------------
	// Create the Position Texture Resource view
	//------------------------------------------
	if( m_pPositionTextureRV == NULL )
	{
		hr = m_pd3dDevice->CreateShaderResourceView( m_pPositionTexture, NULL, &m_pPositionTextureRV );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pPositionTextureRV == NULL )

 	//-----------------------------------------------
	// Create the Position Texture Render Target View
	//-----------------------------------------------
	if( m_pPositionRenderTargetView == NULL )
	{
		hr = m_pd3dDevice->CreateRenderTargetView( m_pPositionTexture, NULL, &m_pPositionRenderTargetView );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pPositionRenderTargetView == NULL )

  	//-----------------------------------
    // Load the 2D Value Gradient Texture
	//-----------------------------------
	if( m_p2DValueGradientTextureRV == NULL )
	{
		D3DX11_IMAGE_INFO valueGradientTextureSrcInfo;
		memset( &valueGradientTextureSrcInfo, 0, sizeof( D3DX11_IMAGE_INFO ) );

		valueGradientTextureSrcInfo.Width				= D3DX11_DEFAULT;
		valueGradientTextureSrcInfo.Height				= D3DX11_DEFAULT;
		valueGradientTextureSrcInfo.Depth				= D3DX11_DEFAULT;
		valueGradientTextureSrcInfo.ArraySize			= D3DX11_DEFAULT;
		valueGradientTextureSrcInfo.MipLevels			= D3DX11_DEFAULT;
		valueGradientTextureSrcInfo.MiscFlags			= D3DX11_DEFAULT;
		valueGradientTextureSrcInfo.Format				= DXGI_FORMAT_R8G8B8A8_UNORM;//D3DX11_DEFAULT;
		valueGradientTextureSrcInfo.ResourceDimension	= D3D11_RESOURCE_DIMENSION_UNKNOWN;
		valueGradientTextureSrcInfo.ImageFileFormat		= D3DX11_IFF_PNG;//D3DX11_DEFAULT;

		D3DX11_IMAGE_LOAD_INFO valueGradientTextureLoadInfo;
		memset( &valueGradientTextureLoadInfo, 0, sizeof( D3DX11_IMAGE_LOAD_INFO ) );

		valueGradientTextureLoadInfo.Width			= D3DX11_DEFAULT;
		valueGradientTextureLoadInfo.Height			= D3DX11_DEFAULT;
		valueGradientTextureLoadInfo.Depth			= D3DX11_DEFAULT;
		valueGradientTextureLoadInfo.FirstMipLevel	= D3DX11_DEFAULT;
		valueGradientTextureLoadInfo.MipLevels		= D3DX11_DEFAULT;
		valueGradientTextureLoadInfo.Usage			= ( D3D11_USAGE )D3DX11_DEFAULT;
		valueGradientTextureLoadInfo.BindFlags		= D3DX11_DEFAULT;
		valueGradientTextureLoadInfo.CpuAccessFlags	= D3DX11_DEFAULT;
		valueGradientTextureLoadInfo.MiscFlags		= D3DX11_DEFAULT;
		valueGradientTextureLoadInfo.Format			= DXGI_FORMAT_R8G8B8A8_UNORM;//DXGI_FORMAT_FROM_FILE;
		valueGradientTextureLoadInfo.Filter			= D3DX11_DEFAULT;
		valueGradientTextureLoadInfo.MipFilter		= D3DX11_DEFAULT;
		valueGradientTextureLoadInfo.pSrcInfo		= &valueGradientTextureSrcInfo;

		hr = D3DX11CreateTextureFromFile( m_pd3dDevice, L"ValueGradientTexture.png", &valueGradientTextureLoadInfo, NULL, &m_p2DValueGradientTextureResource, NULL );
		if( FAILED( hr ) )
		{
			hr = CreateValueGradientTexture( m_p2DValueGradientTexture, NumOctaveWraps );
			if( FAILED( hr ) )
			{
				return hr;

			} // end if( FAILED( hr ) )

			hr = m_pd3dDevice->CreateShaderResourceView( m_p2DValueGradientTexture, NULL, &m_p2DValueGradientTextureRV );
			if( FAILED( hr ) )
			{
				return hr;

			} // end if( FAILED( hr ) )

 		} // end if( FAILED( hr ) )
		else
		{
			hr = m_pd3dDevice->CreateShaderResourceView( m_p2DValueGradientTextureResource, NULL, &m_p2DValueGradientTextureRV );
			if( FAILED( hr ) )
			{
				return hr;

			} // end if( FAILED( hr ) )

		} // end else

	} // end if( m_p2DValueGradientTextureRV == NULL )

	return S_OK;

} // end HRESULT CEngine::InitTextures()

HRESULT CEngine::InitBuffers()
{
	HRESULT hr = S_OK;

	if( m_pd3dDevice == NULL ) return S_FALSE;

 	//----------------------------------
 	// Create the terrain vertex buffers
 	//----------------------------------
	const unsigned int nElementSize = sizeof( RegionVertexElement );
	const unsigned int nMaxBufferSize = 1048576;

	D3D11_BUFFER_DESC regionVertexBufferDesc;
	memset( &regionVertexBufferDesc, 0, sizeof( D3D11_BUFFER_DESC ) );
	regionVertexBufferDesc.Usage			= D3D11_USAGE_STAGING;
	regionVertexBufferDesc.ByteWidth		= nElementSize * nMaxBufferSize;
	regionVertexBufferDesc.CPUAccessFlags	= D3D11_CPU_ACCESS_READ;
	
	for( unsigned int nIndex = 0; nIndex < NumTempVertexBuffers; nIndex++ )
	{
		if( m_pRegionVertexBufferArray[ nIndex ] != NULL ) continue;

		const unsigned int nBufferSize = ( nIndex == 0 ) ? 1 : nMaxBufferSize;

		memset( &regionVertexBufferDesc, 0, sizeof( D3D11_BUFFER_DESC ) );
		regionVertexBufferDesc.Usage		= D3D11_USAGE_DEFAULT;
		regionVertexBufferDesc.ByteWidth	= nElementSize * nBufferSize;
		regionVertexBufferDesc.BindFlags	= D3D11_BIND_VERTEX_BUFFER | ( ( nIndex == 0 ) ? 0 : D3D11_BIND_STREAM_OUTPUT );
		
		hr = m_pd3dDevice->CreateBuffer( &regionVertexBufferDesc, NULL, &m_pRegionVertexBufferArray[ nIndex ] );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

		if( nIndex == 0 )
		{
			RegionVertexElement* pInitialRegionVertices = new RegionVertexElement[ nBufferSize ];
			memset( pInitialRegionVertices, 0, sizeof( RegionVertexElement ) * nBufferSize );

			pInitialRegionVertices[ 0 ].nvPosition.w = 1 << 16; // Initial Radius

			m_pImmediateContext->UpdateSubresource( m_pRegionVertexBufferArray[ nIndex ], 0, NULL, pInitialRegionVertices, 0, 0 );
			SAFE_DELETE( pInitialRegionVertices );

		} // end if( nIndex == 0 )

	} // end for( unsigned int nIndex= 0; nIndex < NumTempVertexBuffers; nIndex++ )

	//-----------------------------------
	// Create the composite vertex buffer
 	//-----------------------------------
	if( m_pCompositeVertexBuffer == NULL )
	{
		const unsigned int nCompositeVertexElementSize = sizeof( CompositeVertexElement );
		const unsigned int nCompositeVertexElementBufferSize = 4;

		D3D11_BUFFER_DESC compositeVertexBufferDesc;
		memset( &compositeVertexBufferDesc, 0, sizeof( D3D11_BUFFER_DESC ) );
		compositeVertexBufferDesc.Usage		= D3D11_USAGE_DEFAULT;
		compositeVertexBufferDesc.ByteWidth	= nCompositeVertexElementSize * nCompositeVertexElementBufferSize;
		compositeVertexBufferDesc.BindFlags	= D3D11_BIND_VERTEX_BUFFER;

		hr = m_pd3dDevice->CreateBuffer( &compositeVertexBufferDesc, NULL, &m_pCompositeVertexBuffer );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

		CompositeVertexElement* pCompositeVertices = new CompositeVertexElement[ nCompositeVertexElementBufferSize ];
		memset( pCompositeVertices, 0, sizeof( CompositeVertexElement ) * nCompositeVertexElementBufferSize );

		pCompositeVertices[ 0 ].fvPosition = D3DXVECTOR4( -1.0f, 1.0f, 1.0f, 1.0f );
		pCompositeVertices[ 0 ].fvTexcoord = D3DXVECTOR2( 0.0f, 0.0f );
		pCompositeVertices[ 1 ].fvPosition = D3DXVECTOR4( 1.0f, 1.0f, 1.0f, 1.0f );
		pCompositeVertices[ 1 ].fvTexcoord = D3DXVECTOR2( 1.0f, 0.0f );
		pCompositeVertices[ 2 ].fvPosition = D3DXVECTOR4( -1.0f, -1.0f, 1.0f, 1.0f );
		pCompositeVertices[ 2 ].fvTexcoord = D3DXVECTOR2( 0.0f, 1.0f );
		pCompositeVertices[ 3 ].fvPosition = D3DXVECTOR4( 1.0f, -1.0f, 1.0f, 1.0f );
		pCompositeVertices[ 3 ].fvTexcoord = D3DXVECTOR2( 1.0f, 1.0f );

		m_pImmediateContext->UpdateSubresource( m_pCompositeVertexBuffer, 0, NULL, pCompositeVertices, 0, 0 );
		SAFE_DELETE( pCompositeVertices );

	} // end if( m_pCompositeVertexBuffer == NULL )

 	//----------------------------------
	// Create the composite index buffer
 	//----------------------------------
	if( m_pCompositeIndexBuffer == NULL )
	{
		D3D11_BUFFER_DESC compositeIndexBufferDesc;
		memset( &compositeIndexBufferDesc, 0, sizeof( D3D11_BUFFER_DESC ) );
		compositeIndexBufferDesc.Usage		= D3D11_USAGE_DEFAULT;
		compositeIndexBufferDesc.ByteWidth	= sizeof( WORD ) * 6;
		compositeIndexBufferDesc.BindFlags	= D3D11_BIND_INDEX_BUFFER;

		WORD indices[] = { 0, 1, 2, 2, 1, 3 };

		D3D11_SUBRESOURCE_DATA indexBufferData;
		memset( &indexBufferData, 0, sizeof( D3D11_SUBRESOURCE_DATA ) );
		indexBufferData.pSysMem = indices;

		hr = m_pd3dDevice->CreateBuffer( &compositeIndexBufferDesc, &indexBufferData, &m_pCompositeIndexBuffer );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pCompositeIndexBuffer == NULL )

 	//---------------------------
	// Create the constant buffer
 	//---------------------------
	if( m_pConstantBuffer == NULL )
	{
		D3D11_BUFFER_DESC cbd1;
		memset( &cbd1, 0, sizeof( D3D11_BUFFER_DESC ) );
		cbd1.ByteWidth		= sizeof( ConstantBuffer );
		cbd1.BindFlags		= D3D11_BIND_CONSTANT_BUFFER;
 
		hr = m_pd3dDevice->CreateBuffer( &cbd1, NULL, &m_pConstantBuffer );
		if( FAILED( hr ) )
		{
			return hr;

		} // end  if( FAILED( hr ) )

	} // end if( m_pConstantBuffer == NULL )

 	//-----------------------------------
	// Create the control constant buffer
 	//-----------------------------------
	if( m_pControlConstantBuffer == NULL )
	{
		D3D11_BUFFER_DESC cbd2;
		memset( &cbd2, 0, sizeof( D3D11_BUFFER_DESC ) );
		cbd2.ByteWidth		= sizeof( D3DXVECTOR4 );
		cbd2.BindFlags		= D3D11_BIND_CONSTANT_BUFFER;
 
		hr = m_pd3dDevice->CreateBuffer( &cbd2, NULL, &m_pControlConstantBuffer );
		if( FAILED( hr ) )
		{
			return hr;

		} // end  if( FAILED( hr ) )

	} // end if( m_pControlConstantBuffer == NULL )

	SetConstants();

	return S_OK;

} // end HRESULT CEngine::InitBuffers()

HRESULT CEngine::InitShaders()
{
	HRESULT hr = S_OK;

	if( m_pd3dDevice == NULL ) return S_FALSE;

 	//-------------------------------------
	// Create the shader #define structures
 	//-------------------------------------
    D3D10_SHADER_MACRO terrainShaderDefines[] =
    {
	    { NULL, NULL },

    }; // end D3D10_SHADER_MACRO terrainShaderDefines[]

	if(    ( m_pRegionPassThroughVertexShader == NULL )
		|| ( m_pRegionVertexLayout == NULL ) )
	{
 		//---------------------------------
		// Compile the region vertex shader
 		//---------------------------------
		ID3DBlob* pRegionPassThroughVSBlob = NULL;
		hr = CompileShaderFromFile( L"./Shader Files/TerrainTessellation.fx", "VS_Region_Pass_Through", "vs_5_0", NULL, &pRegionPassThroughVSBlob );
		if( FAILED( hr ) )
		{
			::MessageBox( NULL, L"VS_Region_Pass_Through cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
			return hr;

		} // end if( FAILED( hr ) )

		if( m_pRegionPassThroughVertexShader == NULL )
		{
			// Create the region vertex shader
			hr = m_pd3dDevice->CreateVertexShader(	pRegionPassThroughVSBlob->GetBufferPointer(), 
													pRegionPassThroughVSBlob->GetBufferSize(), 
													NULL, 
													&m_pRegionPassThroughVertexShader );
			if( FAILED( hr ) )
			{    
				pRegionPassThroughVSBlob->Release();
				return hr;

			} // end if( FAILED( hr ) )

		} // end if( m_pRegionPassThroughVertexShader == NULL )

  		//-------------------------------
		// Define the region input layout
 		//-------------------------------
		if( m_pRegionVertexLayout == NULL )
		{
			D3D11_INPUT_ELEMENT_DESC regionInputLayoutDesc[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_SINT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32B32A32_SINT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },

			}; // end  D3D11_INPUT_ELEMENT_DESC regionInputLayoutDesc[]

			UINT numRegionElements = ARRAYSIZE( regionInputLayoutDesc );

			// Create the region input layout
			hr = m_pd3dDevice->CreateInputLayout(	regionInputLayoutDesc, 
													numRegionElements, 
													pRegionPassThroughVSBlob->GetBufferPointer(),
													pRegionPassThroughVSBlob->GetBufferSize(), 
													&m_pRegionVertexLayout );

		} // end if( m_pRegionVertexLayout == NULL )

		pRegionPassThroughVSBlob->Release();
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( ... )

 	//----------------------------------
	// Compile the split geometry shader
 	//----------------------------------
	if( m_pRegionSplitGeometryShader == NULL )
	{
		ID3DBlob* pSplitRegionGSBlob = NULL;
		hr = CompileShaderFromFile( L"./Shader Files/TerrainTessellation.fx", "GS_Split_Region", "gs_5_0", terrainShaderDefines, &pSplitRegionGSBlob );
		if( FAILED( hr ) )
		{
			::MessageBox( NULL, L"GS_Split_Region cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
			return hr;

		} // end if( FAILED( hr ) )

		// Create the split geometry shader
		D3D11_SO_DECLARATION_ENTRY sodes[] =
		{
			{ 0, "POSITION", 0, 0, 4, 0 },
			{ 0, "TEXCOORD", 0, 0, 4, 0 },
			{ 1, "POSITION", 0, 0, 4, 1 },
			{ 1, "TEXCOORD", 0, 0, 4, 1 },
		};
		UINT numSodesElements = ARRAYSIZE( sodes );
		UINT pSobsArray[] = { sizeof( RegionVertexElement ), sizeof( RegionVertexElement ) };

		hr = m_pd3dDevice->CreateGeometryShaderWithStreamOutput(	pSplitRegionGSBlob->GetBufferPointer(),
																	pSplitRegionGSBlob->GetBufferSize(),
																	sodes,
																	numSodesElements,
																	pSobsArray,
																	2,
																	D3D11_SO_NO_RASTERIZED_STREAM,
																	NULL,
																	&m_pRegionSplitGeometryShader );
		pSplitRegionGSBlob->Release();
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pRegionSplitGeometryShader == NULL )

 	//---------------------------------
	// Compile the face geometry shader
 	//---------------------------------
	if( m_pRegionFaceGeometryShader == NULL )
	{
		ID3DBlob* pRegionFaceGSBlob = NULL;
		hr = CompileShaderFromFile( L"./Shader Files/TerrainTessellation.fx", "GS_Region_Face", "gs_5_0", terrainShaderDefines, &pRegionFaceGSBlob );
		if( FAILED( hr ) )
		{
			::MessageBox( NULL, L"GS_Region_Face cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
			return hr;

		} // end if( FAILED( hr ) )

		// Create the face geometry shader
		hr = m_pd3dDevice->CreateGeometryShader(	pRegionFaceGSBlob->GetBufferPointer(),
													pRegionFaceGSBlob->GetBufferSize(),
													NULL,
													&m_pRegionFaceGeometryShader );
		pRegionFaceGSBlob->Release();
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pRegionFaceGeometryShader == NULL )

 	//--------------------------------------
	// Compile the wireframe geometry shader
 	//--------------------------------------
	if( m_pRegionWireGeometryShader == NULL )
	{
		ID3DBlob* pRegionWireGSBlob = NULL;
		hr = CompileShaderFromFile( L"./Shader Files/TerrainTessellation.fx", "GS_Region_Wireframe", "gs_5_0", terrainShaderDefines, &pRegionWireGSBlob );
		if( FAILED( hr ) )
		{
			::MessageBox( NULL, L"GS_Region_Wireframe cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
			return hr;

		} // end if( FAILED( hr ) )

		// Create the wireframe geometry shader
		hr = m_pd3dDevice->CreateGeometryShader(	pRegionWireGSBlob->GetBufferPointer(),
													pRegionWireGSBlob->GetBufferSize(),
													NULL,
													&m_pRegionWireGeometryShader );
		pRegionWireGSBlob->Release();
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pRegionWireGeometryShader == NULL )

 	//--------------------------------
	// Compile the region pixel shader
 	//--------------------------------
	if( m_pRegionPixelShader == NULL )
	{
		ID3DBlob* pRegionPSBlob = NULL;
		hr = CompileShaderFromFile( L"./Shader Files/TerrainTessellation.fx", "PS_Region", "ps_5_0", terrainShaderDefines, &pRegionPSBlob );
		if( FAILED( hr ) )
		{
			::MessageBox( NULL, L"PS_Region cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
			return hr;

		} // end if( FAILED( hr ) )

		// Create the region pixel shader
		hr = m_pd3dDevice->CreatePixelShader(	pRegionPSBlob->GetBufferPointer(),
												pRegionPSBlob->GetBufferSize(),
												NULL,
												&m_pRegionPixelShader );
		pRegionPSBlob->Release();
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pRegionPixelShader == NULL )

 	//------------------------------------------
	// Compile the region wireframe pixel shader
 	//------------------------------------------
	if( m_pRegionWireframePixelShader == NULL )
	{
		ID3DBlob* pRegionWireframePSBlob = NULL;
		hr = CompileShaderFromFile( L"./Shader Files/TerrainTessellation.fx", "PS_RegionWireframe", "ps_5_0", terrainShaderDefines, &pRegionWireframePSBlob );
		if( FAILED( hr ) )
		{
			::MessageBox( NULL, L"PS_RegionWireframe cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
			return hr;

		} // end if( FAILED( hr ) )

		// Create the region wireframe pixel shader
		hr = m_pd3dDevice->CreatePixelShader(	pRegionWireframePSBlob->GetBufferPointer(),
												pRegionWireframePSBlob->GetBufferSize(),
												NULL,
												&m_pRegionWireframePixelShader );
		pRegionWireframePSBlob->Release();
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pRegionWireframePixelShader == NULL )

	if(    ( m_pCompositeVertexShader == NULL )
		|| ( m_pCompositeVertexLayout == NULL ) )
	{
		ID3DBlob* pCompositeVSBlob = NULL;
		hr = CompileShaderFromFile( L"./Shader Files/TerrainTessellation.fx", "VS_Composite", "vs_5_0", NULL, &pCompositeVSBlob );
		if( FAILED( hr ) )
		{
			::MessageBox( NULL, L"VS_Composite cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
			return hr;

		} // end if( FAILED( hr ) )

  		//------------------------------------
		// Compile the composite vertex shader
 		//------------------------------------
		if( m_pCompositeVertexShader == NULL )
		{
			// Create the composite vertex shader
			hr = m_pd3dDevice->CreateVertexShader(	pCompositeVSBlob->GetBufferPointer(), 
													pCompositeVSBlob->GetBufferSize(), 
													NULL, 
													&m_pCompositeVertexShader );
			if( FAILED( hr ) )
			{    
				pCompositeVSBlob->Release();
				return hr;

			} // end if( FAILED( hr ) )

		} // end if( m_pCompositeVertexShader == NULL )

   		//----------------------------------
		// Define the composite input layout
 		//----------------------------------
		if( m_pCompositeVertexLayout == NULL )
		{
			D3D11_INPUT_ELEMENT_DESC compositeInputLayoutDesc[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT , D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT , D3D11_INPUT_PER_VERTEX_DATA, 0 },

			}; // end  D3D11_INPUT_ELEMENT_DESC compositeInputLayoutDesc[]

			UINT numCompositeElements = ARRAYSIZE( compositeInputLayoutDesc );

			// Create the composite input layout
			hr = m_pd3dDevice->CreateInputLayout(	compositeInputLayoutDesc, 
													numCompositeElements, 
													pCompositeVSBlob->GetBufferPointer(),
													pCompositeVSBlob->GetBufferSize(), 
													&m_pCompositeVertexLayout );

		} // end if( m_pCompositeVertexLayout == NULL )

		pCompositeVSBlob->Release();
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( ... )

 	//------------------------------------
    // Compile the background pixel shader
 	//------------------------------------
	if( m_pBackgroundPixelShader == NULL )
	{
		ID3DBlob* pBackgroundPSBlob = NULL;
		hr = CompileShaderFromFile( L"./Shader Files/TerrainTessellation.fx", "PS_Background", "ps_5_0", terrainShaderDefines, &pBackgroundPSBlob );
		if( FAILED( hr ) )
		{
			::MessageBox( NULL, L"PS_Background cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
			return hr;

		} // end if( FAILED( hr ) )

		// Create the background pixel shader
		hr = m_pd3dDevice->CreatePixelShader(	pBackgroundPSBlob->GetBufferPointer(),
												pBackgroundPSBlob->GetBufferSize(),
												NULL,
												&m_pBackgroundPixelShader );
		pBackgroundPSBlob->Release();
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pBackgroundPixelShader == NULL )

 	//-----------------------------------
    // Compile the composite pixel shader
 	//-----------------------------------
	if( m_pCompositePixelShader == NULL )
	{
		ID3DBlob* pCompositePSBlob = NULL;
		hr = CompileShaderFromFile( L"./Shader Files/TerrainTessellation.fx", "PS_Composite", "ps_5_0", terrainShaderDefines, &pCompositePSBlob );
		if( FAILED( hr ) )
		{
			::MessageBox( NULL, L"PS_Composite cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
			return hr;

		} // end if( FAILED( hr ) )

		// Create the composite pixel shader
		hr = m_pd3dDevice->CreatePixelShader(	pCompositePSBlob->GetBufferPointer(),
												pCompositePSBlob->GetBufferSize(),
												NULL,
												&m_pCompositePixelShader );
		pCompositePSBlob->Release();
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pCompositePixelShader == NULL )

	return S_OK;

} // end HRESULT CEngine::InitShaders()

HRESULT CEngine::InitStateObjects()
{
	HRESULT hr = S_OK;

	if( m_pd3dDevice == NULL ) return S_FALSE;

 	//----------------------------
	// Create the rasterizer state
 	//----------------------------
	if( m_pRasterizerState == NULL )
	{
		D3D11_RASTERIZER_DESC rd;
		memset( &rd, 0, sizeof( D3D11_RASTERIZER_DESC ) );
		rd.FillMode = D3D11_FILL_SOLID;
		rd.CullMode = D3D11_CULL_BACK;
		rd.DepthClipEnable = TRUE;
		rd.AntialiasedLineEnable = FALSE;

		m_pd3dDevice->CreateRasterizerState( &rd, &m_pRasterizerState );
		if( FAILED( hr ) )
		{
			return hr;

		} // end  if( FAILED( hr ) )

	} // end if( m_pRasterizerState == NULL )

 	//--------------------------------------
	// Create the region depth stencil state
 	//--------------------------------------
	if( m_pRegionDepthStencilState == NULL )
	{
		D3D11_DEPTH_STENCIL_DESC regionDepthStencilStateDesc;
		memset( &regionDepthStencilStateDesc, 0, sizeof( D3D11_DEPTH_STENCIL_DESC ) );
		regionDepthStencilStateDesc.DepthEnable		= true;
		regionDepthStencilStateDesc.DepthWriteMask	= D3D11_DEPTH_WRITE_MASK_ALL;
		regionDepthStencilStateDesc.DepthFunc		= D3D11_COMPARISON_LESS_EQUAL;

		m_pd3dDevice->CreateDepthStencilState( &regionDepthStencilStateDesc, &m_pRegionDepthStencilState );
		if( FAILED( hr ) )
		{
			return hr;

		} // end  if( FAILED( hr ) )

	} // end if( m_pRegionDepthStencilState == NULL )

 	//------------------------------------------
	// Create the background depth stencil state
 	//------------------------------------------
	if( m_pBackgroundDepthStencilState == NULL )
	{
		D3D11_DEPTH_STENCIL_DESC backgroundDepthStencilStateDesc;
		memset( &backgroundDepthStencilStateDesc, 0, sizeof( D3D11_DEPTH_STENCIL_DESC ) );
		backgroundDepthStencilStateDesc.DepthEnable		= TRUE;
		backgroundDepthStencilStateDesc.DepthWriteMask	= D3D11_DEPTH_WRITE_MASK_ZERO;
		backgroundDepthStencilStateDesc.DepthFunc		= D3D11_COMPARISON_LESS_EQUAL;

		m_pd3dDevice->CreateDepthStencilState( &backgroundDepthStencilStateDesc, &m_pBackgroundDepthStencilState );
		if( FAILED( hr ) )
		{
			return hr;

		} // end  if( FAILED( hr ) )

	} // end if( m_pBackgroundDepthStencilState == NULL )

 	//-----------------------------------------
	// Create the composite depth stencil state
 	//-----------------------------------------
	if( m_pCompositeDepthStencilState == NULL )
	{
		D3D11_DEPTH_STENCIL_DESC compositeDepthStencilStateDesc;
		memset( &compositeDepthStencilStateDesc, 0, sizeof( D3D11_DEPTH_STENCIL_DESC ) );
		compositeDepthStencilStateDesc.DepthEnable		= TRUE;
		compositeDepthStencilStateDesc.DepthWriteMask	= D3D11_DEPTH_WRITE_MASK_ZERO;
		compositeDepthStencilStateDesc.DepthFunc		= D3D11_COMPARISON_GREATER;

		m_pd3dDevice->CreateDepthStencilState( &compositeDepthStencilStateDesc, &m_pCompositeDepthStencilState );
		if( FAILED( hr ) )
		{
			return hr;

		} // end  if( FAILED( hr ) )

	} // end if( m_pCompositeDepthStencilState == NULL )

 	//------------------------------------
	// Create the linear wrap sample state
 	//------------------------------------
	if( m_pSamplerLinearWrap == NULL )
	{
		D3D11_SAMPLER_DESC linearWrapSampDesc;
		memset( &linearWrapSampDesc, 0, sizeof( D3D11_SAMPLER_DESC ) );
		linearWrapSampDesc.Filter			= D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
		linearWrapSampDesc.AddressU			= D3D11_TEXTURE_ADDRESS_WRAP;
		linearWrapSampDesc.AddressV			= D3D11_TEXTURE_ADDRESS_WRAP;
		linearWrapSampDesc.AddressW			= D3D11_TEXTURE_ADDRESS_WRAP;
		linearWrapSampDesc.ComparisonFunc	= D3D11_COMPARISON_NEVER;
		linearWrapSampDesc.MinLOD			= 0;
		linearWrapSampDesc.MaxLOD			= 0;

		hr = m_pd3dDevice->CreateSamplerState( &linearWrapSampDesc, &m_pSamplerLinearWrap );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pSamplerLinearWrap == NULL )

 	//------------------------------------
	// Create the point clamp sample state
 	//------------------------------------
	if( m_pSamplerPointClamp == NULL )
	{
		D3D11_SAMPLER_DESC pointClampSampDesc;
		memset( &pointClampSampDesc, 0, sizeof( D3D11_SAMPLER_DESC ) );
		pointClampSampDesc.Filter			= D3D11_FILTER_MIN_MAG_MIP_POINT;
		pointClampSampDesc.AddressU			= D3D11_TEXTURE_ADDRESS_CLAMP;
		pointClampSampDesc.AddressV			= D3D11_TEXTURE_ADDRESS_CLAMP;
		pointClampSampDesc.AddressW			= D3D11_TEXTURE_ADDRESS_CLAMP;
		pointClampSampDesc.ComparisonFunc	= D3D11_COMPARISON_NEVER;
		pointClampSampDesc.MinLOD			= 0;
		pointClampSampDesc.MaxLOD			= 0;

		hr = m_pd3dDevice->CreateSamplerState( &pointClampSampDesc, &m_pSamplerPointClamp );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pSamplerPointClamp == NULL )

 	//-----------------------------------
	// Create the point wrap sample state
 	//-----------------------------------
	if( m_pSamplerPointWrap == NULL )
	{
		D3D11_SAMPLER_DESC pointWrapSampDesc;
		memset( &pointWrapSampDesc, 0, sizeof( D3D11_SAMPLER_DESC ) );
		pointWrapSampDesc.Filter			= D3D11_FILTER_MIN_MAG_MIP_POINT;
		pointWrapSampDesc.AddressU			= D3D11_TEXTURE_ADDRESS_WRAP;
		pointWrapSampDesc.AddressV			= D3D11_TEXTURE_ADDRESS_WRAP;
		pointWrapSampDesc.AddressW			= D3D11_TEXTURE_ADDRESS_WRAP;
		pointWrapSampDesc.ComparisonFunc	= D3D11_COMPARISON_NEVER;
		pointWrapSampDesc.MinLOD			= 0;
		pointWrapSampDesc.MaxLOD			= 0;

		hr = m_pd3dDevice->CreateSamplerState( &pointWrapSampDesc, &m_pSamplerPointWrap );
		if( FAILED( hr ) )
		{
			return hr;

		} // end if( FAILED( hr ) )

	} // end if( m_pSamplerPointWrap == NULL )

	return S_OK;

} // end HRESULT CEngine::InitStateObjects()

HRESULT CEngine::CreateValueGradientTexture( ID3D11Texture2D*& p2DValueGradientTexture, const unsigned int nNumOctaveWraps )
{
	CNoise myNoise;
	double fMinNoise = 0.0;
	double fMaxNoise = 0.0;
	double fScale = 2.0;
	int nNumOctaves = 4;
	double fOffset = 0.33;
	double fRotation = 0.3f;
	double fOctaveWrapWeight = 1.0f;

	HRESULT hr = S_OK;

	// Create the 2D Value Gradient Texture
	D3D11_TEXTURE2D_DESC valueGradientTextureDesc;
	memset( &valueGradientTextureDesc, 0, sizeof( D3D11_TEXTURE2D_DESC ) );

	valueGradientTextureDesc.Width				= 512;
	valueGradientTextureDesc.Height				= 512;
	valueGradientTextureDesc.MipLevels			= 1;
	valueGradientTextureDesc.ArraySize			= 1;
	valueGradientTextureDesc.Format				= DXGI_FORMAT_R8G8B8A8_UNORM;//DXGI_FORMAT_R32G32B32A32_FLOAT;
	valueGradientTextureDesc.SampleDesc.Count	= 1;
	valueGradientTextureDesc.SampleDesc.Quality	= 0;
	valueGradientTextureDesc.Usage				= D3D11_USAGE_DEFAULT;
	valueGradientTextureDesc.BindFlags			= D3D11_BIND_SHADER_RESOURCE;
	valueGradientTextureDesc.CPUAccessFlags		= 0;
	valueGradientTextureDesc.MiscFlags			= 0;

	typedef unsigned char rgbaComponentType;
	#define floatToComponentType( x ) ( rgbaComponentType )( ( ( float )( x ) ) * 255.0f )
	#define componentTypeToFloat( x ) ( float )( ( ( float )( x ) ) / 255.0f )

	struct rgbaColor
	{
		rgbaComponentType r;
		rgbaComponentType g;
		rgbaComponentType b;
		rgbaComponentType a;

	}; // end struct rgbaColor

	rgbaColor* pValueGradientTextureData = new rgbaColor[ valueGradientTextureDesc.Width * valueGradientTextureDesc.Height ]; 
	memset( pValueGradientTextureData, 0, sizeof( rgbaColor ) * valueGradientTextureDesc.Width * valueGradientTextureDesc.Height );

	unsigned int x = 0;
	unsigned int y = 0;
	unsigned int z = 0;

	for( int nMode = 0; nMode < 2; nMode++ )
	{
		// Calculate the min max noise values
		for( y = 0; y < valueGradientTextureDesc.Height; y++ )
		{
			for( x = 0; x < valueGradientTextureDesc.Width; x++ )
			{
				unsigned int nIndex = ( y * valueGradientTextureDesc.Width ) + x;

				double fNoise = 0;
				fOctaveWrapWeight = 1.0;
				double fLastWeight = 1.0;

				if( !nNumOctaveWraps )
				{
						fNoise = myNoise.AccumulatingNoise(	( double )x / ( double )valueGradientTextureDesc.Width,
															( double )y / ( double )valueGradientTextureDesc.Height,
															fOffset, fScale, nNumOctaves, fOffset, 0.0, fRotation );

						fNoise = fNoise * 0.5 + 0.5;

				} // end if( !nNumOctaveWraps )
				else
				{
					for( z = 0; z < nNumOctaveWraps; z++ )
					{
						double fOctaveScale = pow( 2.0, ( double )z );
						double fOctaveWrap = myNoise.AccumulatingNoise(	( double )x / ( double )valueGradientTextureDesc.Width,
																		( double )y / ( double )valueGradientTextureDesc.Height,
																		fOffset * fOctaveScale, fScale * fOctaveScale, nNumOctaves, fOffset * fOctaveScale, 0.4 * ( double )z, fRotation );

						fOctaveWrap = abs( fOctaveWrap );
						fOctaveWrap = 1.0f - fOctaveWrap;
						fOctaveWrap = pow( fOctaveWrap, 2.0 );
						fOctaveWrap *= fOctaveWrapWeight;

						fNoise += ( fOctaveWrap * ( 1.0 / fOctaveScale ) );
						fOctaveWrapWeight = fOctaveWrap;
						fOctaveWrapWeight = min( max( 0.0, fOctaveWrapWeight ), 1.0 );
						fLastWeight = fOctaveWrap;

					} // end for( z = 0; z < nNumOctaveWraps; z++ )

				} // end else

				if( !nMode )
				{
					if( !x && !y ) { fMinNoise = fNoise; fMaxNoise = fNoise; }
					fMinNoise = min( fMinNoise, fNoise );
					fMaxNoise = max( fMaxNoise, fNoise );

				} // end if( !nMode )
				else
				{
					fNoise = ( fNoise - fMinNoise ) / ( fMaxNoise - fMinNoise );
					fNoise = min( max( 0.0, fNoise ), 1.0 );

					rgbaComponentType nNoise = floatToComponentType( fNoise );
					pValueGradientTextureData[ nIndex ].a = nNoise;

					float fWeight = ( float )fLastWeight;
					//fWeight = ( fWeight - fMinNoise ) / ( fMaxNoise - fMinNoise );
					fWeight = min( max( 0.0f, fWeight ), 1.0f );
					rgbaComponentType nWeight = floatToComponentType( fWeight );
					pValueGradientTextureData[ nIndex ].b = nWeight;

				} // end else

			} // end for( x = 0; x < valueGradientTextureDesc.Width; x++ )

		} // end for( y = 0; y < valueGradientTextureDesc.Height; y++ )

	} // end for( int nMode = 0; nMode < 2; nMode++ )

	// Calculate the noise gradients
	for( y = 0; y < valueGradientTextureDesc.Height; y++ )
	{
		for( x = 0; x < valueGradientTextureDesc.Width; x++ )
		{
			unsigned int nIndex = ( y * valueGradientTextureDesc.Width ) + x;
			unsigned int nIndexX = ( y * valueGradientTextureDesc.Width ) + ( ( x + 1 ) % valueGradientTextureDesc.Width );
			unsigned int nIndexY = ( ( (y + 1 ) % valueGradientTextureDesc.Height ) * valueGradientTextureDesc.Width ) + x;

			rgbaComponentType nXNoise = pValueGradientTextureData[ nIndexX ].a;
			rgbaComponentType nYNoise = pValueGradientTextureData[ nIndexY ].a;

			pValueGradientTextureData[ nIndex ].r = nXNoise;
			pValueGradientTextureData[ nIndex ].g = nYNoise;

		} // end for( x = 0; x < valueGradientTextureDesc.Width; x++ )

	} // end for( y = 0; y < valueGradientTextureDesc.Height; y++ )

	D3D11_SUBRESOURCE_DATA valueGradientTextureSubresourceData;
	memset( &valueGradientTextureSubresourceData, 0, sizeof( D3D11_SUBRESOURCE_DATA ) );

	valueGradientTextureSubresourceData.SysMemPitch			= valueGradientTextureDesc.Width * sizeof( rgbaColor );
	valueGradientTextureSubresourceData.SysMemSlicePitch	= 0;
	valueGradientTextureSubresourceData.pSysMem				= pValueGradientTextureData;

	hr = m_pd3dDevice->CreateTexture2D( &valueGradientTextureDesc, &valueGradientTextureSubresourceData, &p2DValueGradientTexture );
	if( FAILED( hr ) )
	{
		return hr;

	} // end if( FAILED( hr ) )

	D3DX11SaveTextureToFile( m_pImmediateContext, p2DValueGradientTexture, D3DX11_IFF_PNG, L".\\ValueGradientTexture.png" );

	SAFE_DELETE( pValueGradientTextureData );

	return S_OK;

} // end HRESULT CEngine::CreateValueGradientTexture( ID3D11Texture2D*& p2DValueGradientTexture )

HRESULT CEngine::CompileShaderFromFile(	WCHAR*				szFileName,
										LPCSTR				szEntryPoint,
										LPCSTR				szShaderModel,
										D3D10_SHADER_MACRO* pDefines,
										ID3DBlob**			ppBlobOut )
{
    HRESULT hr = S_OK;

    DWORD dwShaderFlags = 0;

#if defined( DEBUG ) || defined( _DEBUG )
    dwShaderFlags |= D3DCOMPILE_ENABLE_STRICTNESS;
    dwShaderFlags |= D3D10_SHADER_PREFER_FLOW_CONTROL;
    dwShaderFlags |= D3D10_SHADER_OPTIMIZATION_LEVEL3;
    //dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

    ID3DBlob* pErrorBlob = NULL;
    hr = D3DX11CompileFromFile(	szFileName, 
								pDefines, 
								NULL, 
								szEntryPoint, 
								szShaderModel, 
								dwShaderFlags, 
								0, 
								NULL, 
								ppBlobOut, 
								&pErrorBlob, 
								NULL );

    if( FAILED (hr ) )
    {
        if( pErrorBlob != NULL )
		{
            ::OutputDebugStringA( ( char* )pErrorBlob->GetBufferPointer() );
			pErrorBlob->Release();

		} // end if( pErrorBlob != NULL )

        return hr;

    } // end if( FAILED (hr ) )

    if( pErrorBlob )
	{
			pErrorBlob->Release();

	} // end if( pErrorBlob )

    return S_OK;

} // end HRESULT CEngine::CompileShaderFromFile( WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut )

void CEngine::MessageHandler( UINT nMessage, WPARAM wParam, LPARAM lParam )
{
	RECT rc;
    ::GetClientRect( m_hWnd, &rc );
	UINT nWidth = rc.right - rc.left;
	UINT nHeight = rc.bottom - rc.top;

	switch( nMessage )
	{
		case  WM_SIZE:
			ResizeDevice();
			break;

		case WM_MOUSEMOVE:
			{
				if( wParam & MK_LBUTTON )
				{
					int nXPos = ( int )LOWORD( lParam );
					int nYPos = ( int )HIWORD( lParam );

					if( m_orientationInformation.m_bLeftMouseButtonDown )
					{
						m_orientationInformation.m_fvViewRotation[ 0 ] += ( ( float )( nXPos - m_orientationInformation.m_nvPreviousPosition[ 0 ] ) / ( float )( nWidth ) ) * 360.0f;
						m_orientationInformation.m_fvViewRotation[ 1 ] -= ( ( float )( nYPos - m_orientationInformation.m_nvPreviousPosition[ 1 ] ) / ( float )( nHeight ) ) * 360.0f;

					} // end if( m_bLeftMouseButtonDown )
					else
					{
						m_orientationInformation.m_bLeftMouseButtonDown = true;

					} // end else

					m_orientationInformation.m_nvPreviousPosition[ 0 ] = nXPos;
					m_orientationInformation.m_nvPreviousPosition[ 1 ] = nYPos;

				} // end if( wParam & MK_LBUTTON )
				else
				{
					m_orientationInformation.m_bLeftMouseButtonDown = false;

				} // end else

			}
			break;

			case WM_KEYDOWN:
			{
				switch( wParam )
				{
					case VK_LEFT:		m_orientationInformation.m_bLeftArrowButtonDown		= true; break;
					case VK_RIGHT:		m_orientationInformation.m_bRightArrowButtonDown	= true; break;
					case VK_UP:			m_orientationInformation.m_bUpArrowButtonDown		= true; break;
					case VK_DOWN:		m_orientationInformation.m_bDownArrowButtonDown		= true; break;
					case VK_ADD:		m_orientationInformation.m_bAddButtonDown			= true; break;
					case VK_SUBTRACT:	m_orientationInformation.m_bSubtractButtonDown		= true; break;
					case VK_MULTIPLY:	m_orientationInformation.nInverseMaxRegionSpan += 1; break;
					case VK_DIVIDE:		m_orientationInformation.nInverseMaxRegionSpan -= 1; break;
					case VK_PRIOR:		m_orientationInformation.m_bPageUpButtonDown		= true; break;
					case VK_NEXT:		m_orientationInformation.m_bPageDownButtonDown		= true; break;
					case VK_ESCAPE:		SetDefaultSettings(); break;
					case VK_F3:			SetFullScreenMode( !m_bFullScreen ); break;
					case 'h':
					case 'H':			::ShellExecuteA( NULL, "open", "ReadMe.txt", NULL, NULL, SW_SHOW ); break;
					case 'w':
					case 'W':			m_bShowWireframe = !m_bShowWireframe; break;
					case VK_SPACE:		if( !m_bHardwareDevice ) { Render( false ); } break;

				} // end switch( wParam )

			}
			break;

			case WM_KEYUP:
			{
				switch( wParam )
				{
					case VK_LEFT:		m_orientationInformation.m_bLeftArrowButtonDown		= false; break;
					case VK_RIGHT:		m_orientationInformation.m_bRightArrowButtonDown	= false; break;
					case VK_UP:			m_orientationInformation.m_bUpArrowButtonDown		= false; break;
					case VK_DOWN:		m_orientationInformation.m_bDownArrowButtonDown		= false; break;
					case VK_ADD:		m_orientationInformation.m_bAddButtonDown			= false; break;
					case VK_SUBTRACT:	m_orientationInformation.m_bSubtractButtonDown		= false; break;
					case VK_PRIOR:		m_orientationInformation.m_bPageUpButtonDown		= false; break;
					case VK_NEXT:		m_orientationInformation.m_bPageDownButtonDown		= false; break;

				} // end switch( wParam )

			}
			break;

	} // end switch( nMessage )

} // end void CEngine::MessageHandler( ... )

void CEngine::SetDefaultSettings()
{
	SetFullScreenMode( false );
	SetConstants();
	m_bShowWireframe = false;

    RECT rc = { 0, 0, nHSize, nVSize };
    ::AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
	::SetWindowPos( m_hWnd, NULL, 0, 0, rc.right - rc.left, rc.bottom - rc.top, SWP_NOMOVE | SWP_NOZORDER );

} // end void CEngine::SetDefaultSettings()

void CEngine::SetFullScreenMode( bool bFullScreen )
{
	if( m_pSwapChain == NULL ) return;

	if( bFullScreen == TRUE )
	{
		m_pSwapChain->SetFullscreenState( TRUE, NULL );

	} // end if( bFullScreen == TRUE )
	else
	{
		m_pSwapChain->SetFullscreenState( FALSE, NULL );

	} // end else

	m_bFullScreen = bFullScreen;

} // end void CEngine::SetFullScreenMode( bool bFullScreen )

//--------------------------------------------------------------------------------------
// Render a frame
//--------------------------------------------------------------------------------------
void CEngine::Render( bool bHardwareDeviceOnly /* = true */ )
{
	if( bHardwareDeviceOnly && !m_bHardwareDevice ) return;

	LARGE_INTEGER nTimerStart;
	::QueryPerformanceCounter( &nTimerStart );

	UpdateConstants();

    m_pImmediateContext->ClearDepthStencilView( m_pDepthStencilTextureDSV, D3D11_CLEAR_DEPTH, 1.0f, 0 );

	SubdivideRegions();
	RenderRegions();

	CompositeBackground();
	CompositeScene();

	if( m_bShowWireframe )
	{
		RenderWireframeRegions();

	} // end if( m_bShowWireframe )
 
	HRESULT hr = m_pSwapChain->Present( 0, 0 );

	if( hr == S_OK )
	{
		LARGE_INTEGER nTimerEnd;
		::QueryPerformanceCounter( &nTimerEnd );
		LONGLONG nTimerDifference = nTimerEnd.QuadPart - nTimerStart.QuadPart;

		unsigned int nFrameTimeIndex = m_timerInformation.m_nFrameCount % MaxNumStoredFrameTimes;
		m_timerInformation.m_nFrameTimeArray[ nFrameTimeIndex ].QuadPart = nTimerDifference;
		m_timerInformation.m_nFrameCount++;

		if( ( nTimerEnd.QuadPart - m_timerInformation.m_nLastPrintedStatTime.QuadPart ) >= m_timerInformation.m_nTimerFrequency.QuadPart )
		{
			double fTimerFrequency = ( double )m_timerInformation.m_nTimerFrequency.QuadPart;

			LARGE_INTEGER nMinimumTime;
			LARGE_INTEGER nMaximumTime;
			LARGE_INTEGER nTotalTime;

			unsigned int nStoredTimerFrameCount = 0;
			nMinimumTime.QuadPart = 0;
			nMaximumTime.QuadPart = 0;
			nTotalTime.QuadPart = 0;

			CalculateTimerStats( nStoredTimerFrameCount, nMinimumTime, nMaximumTime, nTotalTime );

			if( m_timerInformation.m_nTimerFrequency.QuadPart > 0 )
			{
				if( ( nTotalTime.QuadPart > 0 ) && ( nStoredTimerFrameCount > 0 ) )
				{
					double fAverageTime = ( double )nTotalTime.QuadPart / ( double )nStoredTimerFrameCount;

					char strOutput[ 1024 ];
					sprintf_s(	strOutput,
								1024,
								"TerrainTessellation: Average Frame Render Time = %.2fms (%.2f FPS)  fMaxRegionSpan = %f (Press H for Help)\n",
								( ( fAverageTime / fTimerFrequency ) * 1000.0 ),
								( fTimerFrequency / fAverageTime ),
								m_constantBufferData.fMaxRegionSpan ); 

					::SetWindowTextA( m_hWnd, strOutput );

				} // end if( ( nTotalTime.QuadPart > 0 ) && ( nStoredTimerFrameCount > 0 ) )

			} // if( m_timerInformation.m_nTimerFrequency.QuadPart > 0 ) 
	
			m_timerInformation.m_nLastPrintedStatTime.QuadPart = nTimerEnd.QuadPart;

		} // end if( ... )

	} // end if( hr == S_OK )

} // end void CEngine::Render( ... )

void CEngine::SetConstants()
{
	RECT rc;
    ::GetClientRect( m_hWnd, &rc );
	int nWidth = rc.right - rc.left;
	int nHeight = rc.bottom - rc.top;

	memset( &m_constantBufferData, 0, sizeof( ConstantBuffer ) );

	m_constantBufferData.fvControlPosition.x		 = 0.0f;
	m_constantBufferData.fvControlPosition.y		= fMinEyeHeight;
	m_constantBufferData.fvControlPosition.z		= 0.0f;
	m_orientationInformation.m_fvViewRotation[ 0 ]	= 0.0f;
	m_orientationInformation.m_fvViewRotation[ 1 ]	= 0.0f;

	m_constantBufferData.fvControlPosition.x		= -4.9087896f;
	m_constantBufferData.fvControlPosition.y		= 1.8668327f;
	m_constantBufferData.fvControlPosition.z		= 6.8931317f;
	m_orientationInformation.m_fvViewRotation[ 0 ]	= -540.55469f;
	m_orientationInformation.m_fvViewRotation[ 1 ]	= -19.218750f;

	m_constantBufferData.generalFractalInfo.fractalGeneratorInfo.fLacunarity	= 2.0f * max( 1, NumOctaveWraps );
	m_constantBufferData.generalFractalInfo.fractalGeneratorInfo.fOffset		= 1.0f;
	m_constantBufferData.generalFractalInfo.fractalGeneratorInfo.fGain			= 2.0f;
	m_constantBufferData.generalFractalInfo.fractalGeneratorInfo.fH				= 1.0f;

	float fFrequency = 1.0f;
	for( int nOctaveIndex = 0; nOctaveIndex < 16; nOctaveIndex++ )
	{
		float fRotation = fmod( ( float )nOctaveIndex * 105.0f, 360.0f ) / 180.0f * 3.1415926f;

		m_constantBufferData.generalFractalInfo.fractalOctaveInfo.fSinArray[ nOctaveIndex ]			= sin( fRotation );
		m_constantBufferData.generalFractalInfo.fractalOctaveInfo.fCosArray[ nOctaveIndex ]			= cos( fRotation );
		m_constantBufferData.generalFractalInfo.fractalOctaveInfo.fReverseSinArray[ nOctaveIndex ]	= sin( -fRotation );
		m_constantBufferData.generalFractalInfo.fractalOctaveInfo.fReverseCosArray[ nOctaveIndex ]	= cos( -fRotation );
		m_constantBufferData.generalFractalInfo.fractalOctaveInfo.fXOffsetArray[ nOctaveIndex ]		= fmod( ( float )nOctaveIndex * 1.234567f, 1.0f );
		m_constantBufferData.generalFractalInfo.fractalOctaveInfo.fYOffsetArray[ nOctaveIndex ]		= m_constantBufferData.generalFractalInfo.fractalOctaveInfo.fXOffsetArray[ nOctaveIndex ];
		m_constantBufferData.generalFractalInfo.fractalOctaveInfo.fExponentArray[ nOctaveIndex ]	= pow( fFrequency, -m_constantBufferData.generalFractalInfo.fractalGeneratorInfo.fH );

		fFrequency *= m_constantBufferData.generalFractalInfo.fractalGeneratorInfo.fLacunarity;

	} // end for( int nOctaveIndex = 0; nOctaveIndex < 16; nOctaveIndex++ )

	m_orientationInformation.nInverseMaxRegionSpan = 14;
	m_constantBufferData.fMaxRegionSpan = 1.0f / ( float )m_orientationInformation.nInverseMaxRegionSpan;

	m_constantBufferData.fHSize = ( float )nWidth;
	m_constantBufferData.fVSize = ( float )nHeight;

	UpdateConstants();

} // end void CEngine::SetConstants()

void CEngine::UpdateConstants()
{
	RECT rc;
    ::GetClientRect( m_hWnd, &rc );
	int nWidth = rc.right - rc.left;
	int nHeight = rc.bottom - rc.top;

	if( m_orientationInformation.m_bLeftArrowButtonDown )
	{
		m_orientationInformation.m_fvViewRotation[ 0 ] -= 1.0f;

	} // end if( m_orientationInformation.m_bLeftArrowButtonDown )

	if( m_orientationInformation.m_bRightArrowButtonDown )
	{
		m_orientationInformation.m_fvViewRotation[ 0 ] += 1.0f;

	} // end if( m_orientationInformation.m_bRightArrowButtonDown )

	if( m_orientationInformation.m_bPageUpButtonDown )
	{
		m_orientationInformation.m_fvViewRotation[ 1 ] += 1.0f;

	} // end if( m_orientationInformation.m_bPageUpButtonDown )

	if( m_orientationInformation.m_bPageDownButtonDown )
	{
		m_orientationInformation.m_fvViewRotation[ 1 ] -= 1.0f;

	} // end if( m_orientationInformation.m_bPageDownButtonDown )

	const float fMovementFactor = 1.0f / 50.0f;

	if( m_orientationInformation.m_bAddButtonDown )
	{
		m_constantBufferData.fvControlPosition.y += 0.01f;

	} // end if( m_orientationInformation.m_bAddButtonDown )

	if( m_orientationInformation.m_bSubtractButtonDown )
	{
		m_constantBufferData.fvControlPosition.y -= 0.01f;

	} // end if( m_orientationInformation.m_bSubtractButtonDown )

	m_orientationInformation.nInverseMaxRegionSpan = min( max( 8, m_orientationInformation.nInverseMaxRegionSpan ), 32 );
	m_constantBufferData.fMaxRegionSpan = 1.0f / ( float )m_orientationInformation.nInverseMaxRegionSpan;
	m_constantBufferData.fMaxRegionSpan = min( max( 1.0f / 32.0f, m_constantBufferData.fMaxRegionSpan ), 1.0f / 8.0f );

	if( m_orientationInformation.m_bUpArrowButtonDown )
	{
		D3DXVECTOR3 fvForward( 0.0f, 0.0f, 1.0f );
		D3DXVECTOR3 fvDirection( 0.0f, 0.0f, 1.0f );

		D3DXVec3TransformNormal( &fvDirection, &fvForward, &m_constantBufferData.matView );
		D3DXVec3Normalize( &fvDirection, &fvDirection );
	
		m_constantBufferData.fvControlPosition.x += fvDirection.x * fMovementFactor;
		m_constantBufferData.fvControlPosition.y += fvDirection.y * fMovementFactor;
		m_constantBufferData.fvControlPosition.z += fvDirection.z * fMovementFactor;

	} // end if( m_orientationInformation.m_bUpArrowButtonDown )

	if( m_orientationInformation.m_bDownArrowButtonDown )
	{
		D3DXVECTOR3 fvBackward( 0.0f, 0.0f, -1.0f );
		D3DXVECTOR3 fvDirection( 0.0f, 0.0f, -1.0f );

		D3DXVec3TransformNormal( &fvDirection, &fvBackward, &m_constantBufferData.matView );
		D3DXVec3Normalize( &fvDirection, &fvDirection );

		m_constantBufferData.fvControlPosition.x += fvDirection.x * fMovementFactor;
		m_constantBufferData.fvControlPosition.y += fvDirection.y * fMovementFactor;
		m_constantBufferData.fvControlPosition.z += fvDirection.z * fMovementFactor;

	} // end if( m_orientationInformation.m_bDownArrowButtonDown )

	m_constantBufferData.fvControlPosition.y = max( fMinEyeHeight, m_constantBufferData.fvControlPosition.y );

	float fvViewRotation[ 2 ];
	fvViewRotation[ 0 ] = fmod( fmod( m_orientationInformation.m_fvViewRotation[ 0 ], 360.0f ) + 360.0f, 360.0f );
	fvViewRotation[ 1 ] = fmod( fmod( m_orientationInformation.m_fvViewRotation[ 1 ], 360.0f ) + 360.0f, 360.0f );

	D3DXMATRIX matYRotation;
	D3DXMATRIX matXRotation;
	D3DXMATRIX matRotation;

	D3DXMatrixRotationY( &matYRotation, Radians( fvViewRotation[ 0 ] ) );
	D3DXMatrixRotationX( &matXRotation, Radians( fvViewRotation[ 1 ] ) );
	D3DXMatrixMultiply( &matRotation, &matXRotation, &matYRotation );

	static D3DXVECTOR3 fvEye( 0.0f, fMinEyeHeight, 0.0f ); 
	fvEye.x = m_constantBufferData.fvControlPosition.x;
	fvEye.y = m_constantBufferData.fvControlPosition.y;
	fvEye.z = m_constantBufferData.fvControlPosition.z;

	D3DXMATRIX matEyeToOriginTranslation;
	D3DXMATRIX matOriginToEyeTranslation;
	D3DXMatrixTranslation( &matEyeToOriginTranslation, -fvEye.x, -fvEye.y, -fvEye.z );
	D3DXMatrixTranslation( &matOriginToEyeTranslation, fvEye.x, fvEye.y, fvEye.z );

	D3DXVECTOR3 fvLookAt( 0.0f, 0.0f, -1.0f );
	D3DXVec3TransformNormal( &fvLookAt, &fvLookAt, &matRotation );
	D3DXVec3Normalize( &fvLookAt, &fvLookAt );
	D3DXVec3TransformCoord( &fvLookAt, &fvLookAt, &matOriginToEyeTranslation );

	D3DXVECTOR3 fvUp( 0.0f, 1.0f, 0.0f );
	D3DXVec3TransformNormal( &fvUp, &fvUp, &matRotation );
	D3DXVec3Normalize( &fvUp, &fvUp );

	m_constantBufferData.fvEye = D3DXVECTOR4( fvEye.x, fvEye.y, fvEye.z, 0.0f );
	m_constantBufferData.fvLookAt = D3DXVECTOR4( fvLookAt.x, fvLookAt.y, fvLookAt.z, 0.0f );
	m_constantBufferData.fvUp = D3DXVECTOR4( fvUp.x, fvUp.y, fvUp.z, 0.0f );
	D3DXMatrixLookAtLH( &m_constantBufferData.matView, &fvEye, &fvLookAt, &fvUp );
	D3DXMatrixPerspectiveFovLH( &m_constantBufferData.matProjection, ( 60.0f / 180.0f ) * 3.1415926f, ( float )nWidth / ( float )nHeight, 0.01f, 100.0f );

	D3DXMatrixIdentity( &m_constantBufferData.matWorld );

	D3DXVECTOR3 fvZaxis; D3DXVec3Normalize( &fvZaxis, D3DXVec3Subtract( &fvZaxis, &fvLookAt, &fvEye ) );
	D3DXVECTOR3 fvXaxis; D3DXVec3Normalize( &fvXaxis, D3DXVec3Cross ( &fvXaxis, &fvUp, &fvZaxis ) );
	D3DXVECTOR3 fvYaxis; D3DXVec3Cross( &fvYaxis, &fvZaxis, &fvXaxis );

	m_constantBufferData.matView._11 = fvXaxis.x;
	m_constantBufferData.matView._21 = fvXaxis.y;
	m_constantBufferData.matView._31 = fvXaxis.z;
	m_constantBufferData.matView._41 = -D3DXVec3Dot( &fvXaxis, &fvEye );

	m_constantBufferData.matView._12 = fvYaxis.x;
	m_constantBufferData.matView._22 = fvYaxis.y;
	m_constantBufferData.matView._32 = fvYaxis.z;
	m_constantBufferData.matView._42 = -D3DXVec3Dot( &fvYaxis, &fvEye );

	m_constantBufferData.matView._13 = fvZaxis.x;
	m_constantBufferData.matView._23 = fvZaxis.y;
	m_constantBufferData.matView._33 = fvZaxis.z;
	m_constantBufferData.matView._43 = -D3DXVec3Dot( &fvZaxis, &fvEye );

	m_constantBufferData.matView._14 = 0.0f;
	m_constantBufferData.matView._24 = 0.0f;
	m_constantBufferData.matView._34 = 0.0f;
	m_constantBufferData.matView._44 = 1.0f;

	D3DXMatrixMultiply( &m_constantBufferData.matWorldView, &m_constantBufferData.matWorld, &m_constantBufferData.matView );
	D3DXMatrixMultiply( &m_constantBufferData.matWorldViewProjection, &m_constantBufferData.matWorldView, &m_constantBufferData.matProjection );

    // Left clipping plane
    m_constantBufferData.fvViewFrustumPlanes[ 0 ].x = m_constantBufferData.matWorldViewProjection._14 + m_constantBufferData.matWorldViewProjection._11;
    m_constantBufferData.fvViewFrustumPlanes[ 0 ].y = m_constantBufferData.matWorldViewProjection._24 + m_constantBufferData.matWorldViewProjection._21;
    m_constantBufferData.fvViewFrustumPlanes[ 0 ].z = m_constantBufferData.matWorldViewProjection._34 + m_constantBufferData.matWorldViewProjection._31;
    m_constantBufferData.fvViewFrustumPlanes[ 0 ].w = m_constantBufferData.matWorldViewProjection._44 + m_constantBufferData.matWorldViewProjection._41;
    
    // Right clipping plane
    m_constantBufferData.fvViewFrustumPlanes[ 1 ].x = m_constantBufferData.matWorldViewProjection._14 - m_constantBufferData.matWorldViewProjection._11;
    m_constantBufferData.fvViewFrustumPlanes[ 1 ].y = m_constantBufferData.matWorldViewProjection._24 - m_constantBufferData.matWorldViewProjection._21;
    m_constantBufferData.fvViewFrustumPlanes[ 1 ].z = m_constantBufferData.matWorldViewProjection._34 - m_constantBufferData.matWorldViewProjection._31;
    m_constantBufferData.fvViewFrustumPlanes[ 1 ].w = m_constantBufferData.matWorldViewProjection._44 - m_constantBufferData.matWorldViewProjection._41;
    
    // Top clipping plane
    m_constantBufferData.fvViewFrustumPlanes[ 2 ].x = m_constantBufferData.matWorldViewProjection._14 - m_constantBufferData.matWorldViewProjection._12;
    m_constantBufferData.fvViewFrustumPlanes[ 2 ].y = m_constantBufferData.matWorldViewProjection._24 - m_constantBufferData.matWorldViewProjection._22;
    m_constantBufferData.fvViewFrustumPlanes[ 2 ].z = m_constantBufferData.matWorldViewProjection._34 - m_constantBufferData.matWorldViewProjection._32;
    m_constantBufferData.fvViewFrustumPlanes[ 2 ].w = m_constantBufferData.matWorldViewProjection._44 - m_constantBufferData.matWorldViewProjection._42;
    
    // Bottom clipping plane
    m_constantBufferData.fvViewFrustumPlanes[ 3 ].x = m_constantBufferData.matWorldViewProjection._14 + m_constantBufferData.matWorldViewProjection._12;
    m_constantBufferData.fvViewFrustumPlanes[ 3 ].y = m_constantBufferData.matWorldViewProjection._24 + m_constantBufferData.matWorldViewProjection._22;
    m_constantBufferData.fvViewFrustumPlanes[ 3 ].z = m_constantBufferData.matWorldViewProjection._34 + m_constantBufferData.matWorldViewProjection._32;
    m_constantBufferData.fvViewFrustumPlanes[ 3 ].w = m_constantBufferData.matWorldViewProjection._44 + m_constantBufferData.matWorldViewProjection._42;
    
    // Near clipping plane
    m_constantBufferData.fvViewFrustumPlanes[ 4 ].x = m_constantBufferData.matWorldViewProjection._13;
    m_constantBufferData.fvViewFrustumPlanes[ 4 ].y = m_constantBufferData.matWorldViewProjection._23;
    m_constantBufferData.fvViewFrustumPlanes[ 4 ].z = m_constantBufferData.matWorldViewProjection._33;
    m_constantBufferData.fvViewFrustumPlanes[ 4 ].w = m_constantBufferData.matWorldViewProjection._43;
    
    // Far clipping plane
    m_constantBufferData.fvViewFrustumPlanes[ 5 ].x = m_constantBufferData.matWorldViewProjection._14 - m_constantBufferData.matWorldViewProjection._13;
    m_constantBufferData.fvViewFrustumPlanes[ 5 ].y = m_constantBufferData.matWorldViewProjection._24 - m_constantBufferData.matWorldViewProjection._23;
    m_constantBufferData.fvViewFrustumPlanes[ 5 ].z = m_constantBufferData.matWorldViewProjection._34 - m_constantBufferData.matWorldViewProjection._33;
    m_constantBufferData.fvViewFrustumPlanes[ 5 ].w = m_constantBufferData.matWorldViewProjection._44 - m_constantBufferData.matWorldViewProjection._43;

	D3DXVec4Normalize( &m_constantBufferData.fvViewFrustumPlanes[ 0 ], &m_constantBufferData.fvViewFrustumPlanes[ 0 ] );
	D3DXVec4Normalize( &m_constantBufferData.fvViewFrustumPlanes[ 1 ], &m_constantBufferData.fvViewFrustumPlanes[ 1 ] );
	D3DXVec4Normalize( &m_constantBufferData.fvViewFrustumPlanes[ 2 ], &m_constantBufferData.fvViewFrustumPlanes[ 2 ] );
	D3DXVec4Normalize( &m_constantBufferData.fvViewFrustumPlanes[ 3 ], &m_constantBufferData.fvViewFrustumPlanes[ 3 ] );
	D3DXVec4Normalize( &m_constantBufferData.fvViewFrustumPlanes[ 4 ], &m_constantBufferData.fvViewFrustumPlanes[ 4 ] );
	D3DXVec4Normalize( &m_constantBufferData.fvViewFrustumPlanes[ 5 ], &m_constantBufferData.fvViewFrustumPlanes[ 5 ] );

	D3DXMATRIX matViewProjection;
	D3DXMatrixMultiply( &matViewProjection, &m_constantBufferData.matView, &m_constantBufferData.matProjection );

	// Transpose all of the matrices
	D3DXMatrixTranspose( &m_constantBufferData.matWorld, &m_constantBufferData.matWorld );
	D3DXMatrixTranspose( &m_constantBufferData.matView, &m_constantBufferData.matView );
	D3DXMatrixTranspose( &m_constantBufferData.matProjection, &m_constantBufferData.matProjection );
	D3DXMatrixTranspose( &m_constantBufferData.matWorldView, &m_constantBufferData.matWorldView );
	D3DXMatrixTranspose( &m_constantBufferData.matWorldViewProjection, &m_constantBufferData.matWorldViewProjection );
	
	m_pImmediateContext->UpdateSubresource( m_pConstantBuffer, 0, NULL, &m_constantBufferData, 0, 0 );
	m_pImmediateContext->UpdateSubresource( m_pControlConstantBuffer, 0, NULL, &m_constantBufferData.fvControlPosition, 0, 0 );

} // end void CEngine::UpdateConstants()

void CEngine::CompositeBackground()
{
	ID3D11Buffer*				pCBs[] = { m_pConstantBuffer, m_pControlConstantBuffer };

	ID3D11Buffer*				pNULLBuffer[] = { NULL };
	ID3D11Buffer*				pNULLCBs[] = { NULL, NULL };

    UINT nCompositeVertexElementStride =  sizeof( CompositeVertexElement );
    UINT nCompositeVertexElementOffset = 0;

    // Set the input layout and primitive topology
    m_pImmediateContext->IASetInputLayout( m_pCompositeVertexLayout );
	m_pImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
	m_pImmediateContext->IASetVertexBuffers( 0, 1, &m_pCompositeVertexBuffer, &nCompositeVertexElementStride, &nCompositeVertexElementOffset );
	m_pImmediateContext->IASetIndexBuffer( m_pCompositeIndexBuffer, DXGI_FORMAT_R16_UINT, 0 );

	// Set active shaders, constants and resources
	m_pImmediateContext->VSSetShader( m_pCompositeVertexShader, NULL, 0 );
	m_pImmediateContext->PSSetShader( m_pBackgroundPixelShader, NULL, 0 );
	m_pImmediateContext->VSSetConstantBuffers( 0, 2, pCBs );
	m_pImmediateContext->PSSetConstantBuffers( 0, 2, pCBs );

	// Set the render targets
    m_pImmediateContext->OMSetRenderTargets( 1, &m_pRenderTargetView, m_pDepthStencilTextureDSV );
	m_pImmediateContext->OMSetDepthStencilState( m_pBackgroundDepthStencilState, 0 );

	m_pImmediateContext->DrawIndexed( 6, 0, 0 );

    // Reset the input layout and primitive topology
    m_pImmediateContext->IASetInputLayout( NULL );
	m_pImmediateContext->IASetPrimitiveTopology( D3D_PRIMITIVE_TOPOLOGY_UNDEFINED );
	m_pImmediateContext->IASetVertexBuffers( 0, 1, pNULLBuffer, &nCompositeVertexElementStride, &nCompositeVertexElementOffset );
	m_pImmediateContext->IASetIndexBuffer( NULL, DXGI_FORMAT_UNKNOWN, 0 );

	// Reset active shaders, constants and resources
	m_pImmediateContext->VSSetShader( NULL, NULL, 0 );
    m_pImmediateContext->PSSetShader( NULL, NULL, 0 );
	m_pImmediateContext->VSSetConstantBuffers( 0, 2, pNULLCBs );
	m_pImmediateContext->PSSetConstantBuffers( 0, 2, pNULLCBs );

	// Reset the render targets
    m_pImmediateContext->OMSetRenderTargets( 0, NULL, NULL );
	m_pImmediateContext->OMSetDepthStencilState( NULL, 0 );

} // end void CEngine::CompositeBackground()

void CEngine::SubdivideRegions()
{
	ID3D11ShaderResourceView*	pGSRVs[] = { m_p2DValueGradientTextureRV };
	ID3D11SamplerState*			pGSSs[] = {	m_pSamplerLinearWrap,
											m_pSamplerPointClamp,
											m_pSamplerPointWrap };
	ID3D11Buffer*				pCBs[] = { m_pConstantBuffer, m_pControlConstantBuffer };

	ID3D11Buffer*				pNULLSOBuffers[] = { NULL, NULL };
	ID3D11ShaderResourceView*	pNULLRVs[] = { NULL };
	ID3D11SamplerState*			pNULLSSs[] = { NULL, NULL, NULL };
	UINT						pNULLBufferOffsets[] = { 0, 0 };
	ID3D11Buffer*				pNULLCBs[] = { NULL, NULL };

    UINT nRegionVertexElementStride =  sizeof( RegionVertexElement );
    UINT nRegionVertexElementOffset = 0;

    // Set the input layout and primitive topology
    m_pImmediateContext->IASetInputLayout( m_pRegionVertexLayout );
	m_pImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_POINTLIST );

	// Set active shaders, constants and resources
	m_pImmediateContext->VSSetShader( m_pRegionPassThroughVertexShader, NULL, 0 );
    m_pImmediateContext->GSSetShader( m_pRegionSplitGeometryShader, NULL, 0 );
	m_pImmediateContext->GSSetConstantBuffers( 0, 2, pCBs );
	m_pImmediateContext->GSSetShaderResources( 0, 1, pGSRVs );
	m_pImmediateContext->GSSetSamplers( 0, 3, pGSSs );
    m_pImmediateContext->PSSetShader( NULL, NULL, 0 );

	const unsigned int nMaxNumSplits = 16;
	for( unsigned int nSplitIndex = 0; nSplitIndex < nMaxNumSplits; nSplitIndex++ )
	{
		m_constantBufferData.fvControlPosition[ 3 ] = ( float )nSplitIndex / ( float )( nMaxNumSplits - 1 );
		if( nSplitIndex == ( nMaxNumSplits - 1 ) ) m_constantBufferData.fvControlPosition[ 3 ] = 1.0f;
		m_pImmediateContext->UpdateSubresource( m_pControlConstantBuffer, 0, NULL, &m_constantBufferData.fvControlPosition, 0, 0 );

		unsigned int nInputBufferIndex = ( nSplitIndex == 0 ) ? InitialBufferIndex : ( ( nSplitIndex % 2 ) + TempStartBufferIndex );
		unsigned int nOutputBufferIndex = ( ( nSplitIndex + 1 ) % 2 ) + TempStartBufferIndex;

		m_pImmediateContext->IASetVertexBuffers( 0, 1, &( m_pRegionVertexBufferArray[ nInputBufferIndex ] ), &nRegionVertexElementStride, &nRegionVertexElementOffset );

		ID3D11Buffer* pSOBuffers[] = { m_pRegionVertexBufferArray[ nOutputBufferIndex ], m_pRegionVertexBufferArray[ FinalBufferIndex ] };
		UINT pBufferOffsets[] = { 0, ( nSplitIndex == 0 ) ? 0 : -1 };
		m_pImmediateContext->SOSetTargets( 2, pSOBuffers, pBufferOffsets );
		
		if( nSplitIndex == 0 )
		{
			m_pImmediateContext->Draw( 1, 0 );

		} // end if( nSplitIndex == 0 )
		else
		{
			m_pImmediateContext->DrawAuto();

		} // end else

		m_pImmediateContext->IASetVertexBuffers( 0, 1, pNULLSOBuffers, &nRegionVertexElementStride, &nRegionVertexElementOffset );
		m_pImmediateContext->SOSetTargets( 2, pNULLSOBuffers, pNULLBufferOffsets );

	} // end for( unsigned int nSplitIndex = 0; nSplitIndex < nNumSplits; nSplitIndex++ )

    // Reset the input layout and primitive topology
    m_pImmediateContext->IASetInputLayout( NULL );
	m_pImmediateContext->IASetPrimitiveTopology( D3D_PRIMITIVE_TOPOLOGY_UNDEFINED );

	// Reset active shaders, constants and resources
	m_pImmediateContext->VSSetShader( NULL, NULL, 0 );
    m_pImmediateContext->GSSetShader( NULL, NULL, 0 );
	m_pImmediateContext->GSSetConstantBuffers( 0, 2, pNULLCBs );
	m_pImmediateContext->GSSetShaderResources( 0, 1, pNULLRVs );
	m_pImmediateContext->GSSetSamplers( 0, 3, pNULLSSs );
    m_pImmediateContext->PSSetShader( NULL, NULL, 0 );

} // end void CEngine::SubdivideRegions()

void CEngine::RenderRegions()
{
    float fvClearColor[ 4 ] = { 0.0f, 0.0f, 0.0f, 0.0f };
    m_pImmediateContext->ClearRenderTargetView( m_pPositionRenderTargetView, fvClearColor );

	ID3D11ShaderResourceView*	pGSRVs[] = { m_p2DValueGradientTextureRV };
	ID3D11SamplerState*			pGSSs[] = {	m_pSamplerLinearWrap,
											m_pSamplerPointClamp,
											m_pSamplerPointWrap };
	ID3D11Buffer*				pCBs[] = { m_pConstantBuffer, m_pControlConstantBuffer };
	ID3D11RenderTargetView*		pOMRTs[] = { m_pPositionRenderTargetView };

	ID3D11Buffer*				pNULLBuffer[] = { NULL };
	ID3D11ShaderResourceView*	pNULLRVs[] = { NULL };
	ID3D11SamplerState*			pNULLSSs[] = { NULL, NULL, NULL };
	ID3D11Buffer*				pNULLCBs[] = { NULL, NULL };

	unsigned int nBufferIndex = FinalBufferIndex;
    UINT nRegionVertexElementStride =  sizeof( RegionVertexElement );
    UINT nRegionVertexElementOffset = 0;

    // Set the input layout and primitive topology
    m_pImmediateContext->IASetInputLayout( m_pRegionVertexLayout );
	m_pImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_POINTLIST );
	m_pImmediateContext->IASetVertexBuffers( 0, 1, &m_pRegionVertexBufferArray[ nBufferIndex ], &nRegionVertexElementStride, &nRegionVertexElementOffset );
	m_pImmediateContext->RSSetState( m_pRasterizerState );

	// Set active shaders, constants and resources
	m_pImmediateContext->VSSetShader( m_pRegionPassThroughVertexShader, NULL, 0 );
	m_pImmediateContext->GSSetShader( m_pRegionFaceGeometryShader, NULL, 0 );
	m_pImmediateContext->GSSetConstantBuffers( 0, 2, pCBs );
	m_pImmediateContext->GSSetShaderResources( 0, 1, pGSRVs );
	m_pImmediateContext->GSSetSamplers( 0, 3, pGSSs );
	m_pImmediateContext->PSSetShader( m_pRegionPixelShader, NULL, 0 );
	m_pImmediateContext->PSSetConstantBuffers( 0, 2, pCBs );

	// Set the render targets
    m_pImmediateContext->OMSetRenderTargets( 1, pOMRTs, m_pDepthStencilTextureDSV );
	m_pImmediateContext->OMSetDepthStencilState( m_pRegionDepthStencilState, 0 );

	if( nBufferIndex == InitialBufferIndex )
	{
		m_pImmediateContext->Draw( 1, 0 );

	} // end if( nBufferIndex == InitialBufferIndex )
	else
	{
		m_pImmediateContext->DrawAuto();

	} // end else

    // Reset the input layout and primitive topology
    m_pImmediateContext->IASetInputLayout( NULL );
	m_pImmediateContext->IASetPrimitiveTopology( D3D_PRIMITIVE_TOPOLOGY_UNDEFINED );
	m_pImmediateContext->IASetVertexBuffers( 0, 1, pNULLBuffer, &nRegionVertexElementStride, &nRegionVertexElementOffset );
	m_pImmediateContext->RSSetState( NULL );

	// Reset active shaders, constants and resources
	m_pImmediateContext->VSSetShader( NULL, NULL, 0 );
    m_pImmediateContext->GSSetShader( NULL, NULL, 0 );
    m_pImmediateContext->PSSetShader( NULL, NULL, 0 );
	m_pImmediateContext->GSSetConstantBuffers( 0, 2, pNULLCBs );
	m_pImmediateContext->GSSetShaderResources( 0, 1, pNULLRVs );
	m_pImmediateContext->GSSetSamplers( 0, 3, pNULLSSs );
	m_pImmediateContext->PSSetConstantBuffers( 0, 2, pNULLCBs );

	// Reset the render targets
    m_pImmediateContext->OMSetRenderTargets( 0, NULL, NULL );
	m_pImmediateContext->OMSetDepthStencilState( NULL, 0 );

} // end void CEngine::RenderRegions()

void CEngine::CompositeScene()
{
	ID3D11ShaderResourceView*	pPSRVs[] = {	m_p2DValueGradientTextureRV, 
												m_pPositionTextureRV };
	ID3D11SamplerState*			pPSSs[] = {	m_pSamplerLinearWrap,
											m_pSamplerPointClamp,
											m_pSamplerPointWrap };
	ID3D11Buffer*				pCBs[] = { m_pConstantBuffer, m_pControlConstantBuffer };

	ID3D11Buffer*				pNULLBuffer[] = { NULL };
	ID3D11ShaderResourceView*	pNULLRVs[] = { NULL, NULL };
	ID3D11SamplerState*			pNULLSSs[] = { NULL, NULL, NULL };
	ID3D11Buffer*				pNULLCBs[] = { NULL, NULL };

    UINT nCompositeVertexElementStride =  sizeof( CompositeVertexElement );
    UINT nCompositeVertexElementOffset = 0;

    // Set the input layout and primitive topology
    m_pImmediateContext->IASetInputLayout( m_pCompositeVertexLayout );
	m_pImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
	m_pImmediateContext->IASetVertexBuffers( 0, 1, &m_pCompositeVertexBuffer, &nCompositeVertexElementStride, &nCompositeVertexElementOffset );
	m_pImmediateContext->IASetIndexBuffer( m_pCompositeIndexBuffer, DXGI_FORMAT_R16_UINT, 0 );

	// Set active shaders, constants and resources
	m_pImmediateContext->VSSetShader( m_pCompositeVertexShader, NULL, 0 );
	m_pImmediateContext->PSSetShader( m_pCompositePixelShader, NULL, 0 );
	m_pImmediateContext->VSSetConstantBuffers( 0, 2, pCBs );
	m_pImmediateContext->PSSetConstantBuffers( 0, 2, pCBs );
	m_pImmediateContext->PSSetShaderResources( 0, 2, pPSRVs );
	m_pImmediateContext->PSSetSamplers( 0, 3, pPSSs );

	// Set the render targets
    m_pImmediateContext->OMSetRenderTargets( 1, &m_pRenderTargetView, m_pDepthStencilTextureDSV );
	m_pImmediateContext->OMSetDepthStencilState( m_pCompositeDepthStencilState, 0 );

	m_pImmediateContext->DrawIndexed( 6, 0, 0 );

    // Reset the input layout and primitive topology
    m_pImmediateContext->IASetInputLayout( NULL );
	m_pImmediateContext->IASetPrimitiveTopology( D3D_PRIMITIVE_TOPOLOGY_UNDEFINED );
	m_pImmediateContext->IASetVertexBuffers( 0, 1, pNULLBuffer, &nCompositeVertexElementStride, &nCompositeVertexElementOffset );
	m_pImmediateContext->IASetIndexBuffer( NULL, DXGI_FORMAT_UNKNOWN, 0 );

	// Reset active shaders, constants and resources
	m_pImmediateContext->VSSetShader( NULL, NULL, 0 );
    m_pImmediateContext->PSSetShader( NULL, NULL, 0 );
	m_pImmediateContext->VSSetConstantBuffers( 0, 2, pNULLCBs );
	m_pImmediateContext->PSSetConstantBuffers( 0, 2, pNULLCBs );
	m_pImmediateContext->PSSetShaderResources( 0, 2, pNULLRVs );
	m_pImmediateContext->PSSetSamplers( 0, 3, pNULLSSs );

	// Reset the render targets
    m_pImmediateContext->OMSetRenderTargets( 0, NULL, NULL );
	m_pImmediateContext->OMSetDepthStencilState( NULL, 0 );

} // end void CEngine::CompositeScene()

void CEngine::RenderWireframeRegions()
{
	ID3D11ShaderResourceView*	pPSRVs[] = {	m_p2DValueGradientTextureRV,
												m_pPositionTextureRV };
	ID3D11SamplerState*			pPSSs[] = {	m_pSamplerLinearWrap,
											m_pSamplerPointClamp,
											m_pSamplerPointWrap };
	ID3D11Buffer*				pCBs[] = { m_pConstantBuffer, m_pControlConstantBuffer };

	ID3D11Buffer*				pNULLBuffer[] = { NULL };
	ID3D11ShaderResourceView*	pNULLRVs[] = { NULL, NULL };
	ID3D11SamplerState*			pNULLSSs[] = { NULL, NULL, NULL, NULL };
	ID3D11Buffer*				pNULLCBs[] = { NULL, NULL };

	unsigned int nBufferIndex = FinalBufferIndex;
    UINT nRegionVertexElementStride =  sizeof( RegionVertexElement );
    UINT nRegionVertexElementOffset = 0;

    // Set the input layout and primitive topology
    m_pImmediateContext->IASetInputLayout( m_pRegionVertexLayout );
	m_pImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_POINTLIST );
	m_pImmediateContext->IASetVertexBuffers( 0, 1, &m_pRegionVertexBufferArray[ nBufferIndex ], &nRegionVertexElementStride, &nRegionVertexElementOffset );
	m_pImmediateContext->RSSetState( m_pRasterizerState );

	// Set active shaders, constants and resources
	m_pImmediateContext->VSSetShader( m_pRegionPassThroughVertexShader, NULL, 0 );
	m_pImmediateContext->GSSetShader( m_pRegionFaceGeometryShader, NULL, 0 );
	m_pImmediateContext->GSSetConstantBuffers( 0, 2, pCBs );
	m_pImmediateContext->GSSetShaderResources( 0, 2, pPSRVs );
	m_pImmediateContext->GSSetSamplers( 0, 3, pPSSs );
	m_pImmediateContext->PSSetShader( m_pRegionWireframePixelShader, NULL, 0 );
	m_pImmediateContext->PSSetConstantBuffers( 0, 2, pCBs );
	m_pImmediateContext->PSSetShaderResources( 0, 2, pPSRVs );
	m_pImmediateContext->PSSetSamplers( 0, 3, pPSSs );

	// Set the render targets
    m_pImmediateContext->OMSetRenderTargets( 1, &m_pRenderTargetView, m_pDepthStencilTextureDSV );

	m_pImmediateContext->GSSetShader( m_pRegionWireGeometryShader, NULL, 0 );

	if( nBufferIndex == InitialBufferIndex )
	{
		m_pImmediateContext->Draw( 1, 0 );

	} // end if( nBufferIndex == InitialBufferIndex )
	else
	{
		m_pImmediateContext->DrawAuto();

	} // end else

    // Reset the input layout and primitive topology
    m_pImmediateContext->IASetInputLayout( NULL );
	m_pImmediateContext->IASetPrimitiveTopology( D3D_PRIMITIVE_TOPOLOGY_UNDEFINED );
	m_pImmediateContext->IASetVertexBuffers( 0, 1, pNULLBuffer, &nRegionVertexElementStride, &nRegionVertexElementOffset );
	m_pImmediateContext->RSSetState( NULL );

	// Reset active shaders, constants and resources
	m_pImmediateContext->VSSetShader( NULL, NULL, 0 );
    m_pImmediateContext->GSSetShader( NULL, NULL, 0 );
	m_pImmediateContext->GSSetConstantBuffers( 0, 2, pNULLCBs );
	m_pImmediateContext->GSSetShaderResources( 0, 2, pNULLRVs );
	m_pImmediateContext->GSSetSamplers( 0, 3, pNULLSSs );
    m_pImmediateContext->PSSetShader( NULL, NULL, 0 );
	m_pImmediateContext->PSSetConstantBuffers( 0, 2, pNULLCBs );
	m_pImmediateContext->PSSetShaderResources( 0, 2, pNULLRVs );
	m_pImmediateContext->PSSetSamplers( 0, 3, pNULLSSs );

	// Reset the render targets
    m_pImmediateContext->OMSetRenderTargets( 0, NULL, NULL );

} // end void CEngine::RenderWireframeRegions()

void CEngine::CalculateTimerStats(	unsigned int &nStoredTimerFrameCount,
									LARGE_INTEGER &nMinimumTime,
									LARGE_INTEGER &nMaximumTime,
									LARGE_INTEGER &nTotalTime )
{
	nStoredTimerFrameCount = min( MaxNumStoredFrameTimes, m_timerInformation.m_nFrameCount );

	nMinimumTime.QuadPart = 0;
	nMaximumTime.QuadPart = 0;
	nTotalTime.QuadPart = 0;

	for( unsigned int nFrameTimeIndex = 0; nFrameTimeIndex < nStoredTimerFrameCount; nFrameTimeIndex++ )
	{
		if( nFrameTimeIndex == 0 )
		{
			nMinimumTime.QuadPart = m_timerInformation.m_nFrameTimeArray[ nFrameTimeIndex ].QuadPart;
			nMaximumTime.QuadPart = m_timerInformation.m_nFrameTimeArray[ nFrameTimeIndex ].QuadPart;
			nTotalTime.QuadPart = m_timerInformation.m_nFrameTimeArray[ nFrameTimeIndex ].QuadPart;

		} // end if( nFrameTimeIndex == 0 )
		else
		{
			nMinimumTime.QuadPart = min( nMinimumTime.QuadPart, m_timerInformation.m_nFrameTimeArray[ nFrameTimeIndex ].QuadPart );
			nMaximumTime.QuadPart = max( nMaximumTime.QuadPart, m_timerInformation.m_nFrameTimeArray[ nFrameTimeIndex ].QuadPart );
			nTotalTime.QuadPart += m_timerInformation.m_nFrameTimeArray[ nFrameTimeIndex ].QuadPart;

		} // end else

	} // end for( unsigned int nFrameTimeIndex = 0; nFrameTimeIndex < nStoredTimerFrameCount; nFrameTimeIndex++ )

}; // end void CEngine::CalculateTimerStats( ... )