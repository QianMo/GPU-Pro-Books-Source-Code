// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

#include "D3D11Object.h"

#ifdef WINDOWS

D3D11Object::D3D11Object( void )
{
	m_pDevice = NULL;
	m_pDeviceContext = NULL;
	m_pSwapChain = NULL;
	m_pRenderTargetView = NULL;
	m_pBackBufferTexture = NULL;
	m_pSamplerLinear = NULL;
}

D3D11Object::~D3D11Object( void )
{
	SAFE_RELEASE( m_pDevice );
	SAFE_RELEASE( m_pDeviceContext );
	SAFE_RELEASE( m_pSwapChain );
	SAFE_RELEASE( m_pRenderTargetView );
	SAFE_RELEASE( m_pBackBufferTexture );
	SAFE_RELEASE( m_pSamplerLinear );
}

HRESULT D3D11Object::Initialize( HWND &hWnd, ID3D11UnorderedAccessView** ppResultUAV )
{
	printf("Initializing DXObject...\n");

	HRESULT hr = S_OK;

	RECT rc;
	GetClientRect( hWnd, &rc );
	UINT width = rc.right - rc.left;
	UINT height = rc.bottom - rc.top;

	InitDeviceContext(static_cast<int>(HEIGHT), static_cast<int>(WIDTH), hWnd );
	m_pSwapChain->GetBuffer( 0, __uuidof( ID3D11Texture2D ), reinterpret_cast<void**>(&m_pBackBufferTexture) );
	CreateBufferTexUAV( m_pBackBufferTexture, ppResultUAV );
	m_pDevice->CreateRenderTargetView( m_pBackBufferTexture, NULL, &m_pRenderTargetView );
	CreateViewport(width, height);
	CreateSamplerStates();

	return hr;
}

HRESULT D3D11Object::CreateSamplerStates()
{
	HRESULT hr = S_OK;

	LoadSamplerState(&m_pSamplerLinear);
	GetDeviceContext()->CSSetSamplers( 0, 1, &m_pSamplerLinear );

	return hr;
}

//--------------------------------------------------------------------------------------
// Helper function to compile an hlsl shader from file, 
// its binary compiled code is returned
//--------------------------------------------------------------------------------------
HRESULT D3D11Object::CompileShaderFromFile( WCHAR* szFileName, LPCSTR szEntryPoint, 
LPCSTR szShaderModel, ID3DBlob** ppBlobOut )
{
    HRESULT hr = S_OK;

    // find the file
    //WCHAR str[MAX_PATH];
    //V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, szFileName ) );
	WCHAR *str = szFileName;
    // open the file 
	//"createFile" creates or opens the file
    HANDLE hFile = CreateFile( szFileName, GENERIC_READ, FILE_SHARE_READ, NULL, 
        OPEN_EXISTING,
        FILE_FLAG_SEQUENTIAL_SCAN, NULL );
    if( INVALID_HANDLE_VALUE == hFile )
        return E_FAIL;

    // Get the file size
    LARGE_INTEGER FileSize;
    GetFileSizeEx( hFile, &FileSize );

    // create enough space for the file data
    BYTE* pFileData = new BYTE[ FileSize.LowPart ];
    if( !pFileData )
        return E_OUTOFMEMORY;

    // read the data in
    DWORD BytesRead;
    if( !ReadFile( hFile, pFileData, FileSize.LowPart, &BytesRead, NULL ) )
        return E_FAIL; 

    CloseHandle( hFile );

    // Compile the shader
    char pFilePathName[MAX_PATH];        
    WideCharToMultiByte(CP_ACP, 0, str, -1, pFilePathName, MAX_PATH, NULL, NULL);
    ID3DBlob* pErrorBlob;
    hr = D3DCompile( pFileData, FileSize.LowPart, pFilePathName, NULL, NULL, szEntryPoint, 
                      szShaderModel, D3D10_SHADER_ENABLE_STRICTNESS, 0, ppBlobOut, &pErrorBlob );


    delete []pFileData;
  
    if( FAILED(hr) )
    {
        printf( (char*)pErrorBlob->GetBufferPointer() );
        SAFE_RELEASE( pErrorBlob );
        return hr;
    }
    SAFE_RELEASE( pErrorBlob );

    return S_OK;
}

HRESULT D3D11Object::CreateBufferTexUAV( ID3D11Texture2D* pTexture, ID3D11UnorderedAccessView** ppUAVOut )
{
	D3D11_UNORDERED_ACCESS_VIEW_DESC DescUAV;
	ZeroMemory( &DescUAV, sizeof(DescUAV) );
	DescUAV.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	DescUAV.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
	DescUAV.Texture2D.MipSlice=0;

	return m_pDevice->CreateUnorderedAccessView( pTexture, &DescUAV, ppUAVOut );
}

//--------------------------------------------------------------------------------------
// Create a CPU accessible buffer and download the content of a GPU buffer into it
// This function is very useful for debugging CS programs
//-------------------------------------------------------------------------------------- 
ID3D11Buffer* D3D11Object::CreateAndCopyToDebugBuf( ID3D11Buffer* pBuffer )
{
    ID3D11Buffer* debugbuf = NULL;

    D3D11_BUFFER_DESC desc;
    ZeroMemory( &desc, sizeof(desc) );
    pBuffer->GetDesc( &desc );
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.MiscFlags = 0;
    if ( SUCCEEDED(m_pDevice->CreateBuffer(&desc, NULL, &debugbuf)) )
    {
		m_pDeviceContext->CopyResource( debugbuf, pBuffer );
    }

    return debugbuf;
}

void D3D11Object::CreateViewport( UINT uiWidth, UINT uiHeight )
{
	printf("Create viewport\n");
    D3D11_VIEWPORT vp;
    vp.Width = static_cast<float>(uiWidth);
    vp.Height = static_cast<float>(uiHeight);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    m_pDeviceContext->RSSetViewports( 1, &vp );
}

//--------------------------------------------------------------------------------------
// Update Time
//--------------------------------------------------------------------------------------
void D3D11Object::UpdateTime(float& fTime)
{
    static DWORD dwTimeStart = 0;
    DWORD dwTimeCur = GetTickCount();
    if( dwTimeStart == 0 )
		dwTimeStart = dwTimeCur;
	fTime = ( dwTimeCur - dwTimeStart ) / 1000.0f;
}

void D3D11Object::InitDeviceContext( int iHeight, int iWidth, HWND& hWnd )
{
	printf("Init device context...\n");

	HRESULT hr;

	D3D_FEATURE_LEVEL levelsWanted[] = 
	{ 
		D3D_FEATURE_LEVEL_11_0, 
		D3D_FEATURE_LEVEL_10_1, 
		D3D_FEATURE_LEVEL_10_0,
	};
	D3D_FEATURE_LEVEL featureLevel;
	
	UINT numLevelsWanted = sizeof( levelsWanted ) / sizeof( levelsWanted[0] );

	D3D_DRIVER_TYPE driverTypes[] =
	{
		D3D_DRIVER_TYPE_HARDWARE,
		D3D_DRIVER_TYPE_REFERENCE,
	};
	D3D_DRIVER_TYPE driverType;
	UINT numDriverTypes = sizeof( driverTypes ) / sizeof( driverTypes[0] );

	DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory( &sd, sizeof( sd ) );
    sd.BufferCount = 1;
    sd.BufferDesc.Width = WIDTH;
    sd.BufferDesc.Height = HEIGHT;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_UNORDERED_ACCESS;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;

	for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )
	{
		driverType = driverTypes[driverTypeIndex];
		UINT createDeviceFlags = NULL;

		hr = D3D11CreateDeviceAndSwapChain( NULL, driverType, NULL, createDeviceFlags, levelsWanted, numLevelsWanted,
                                          D3D11_SDK_VERSION, &sd, &m_pSwapChain, &m_pDevice, &featureLevel, &m_pDeviceContext );

		if( SUCCEEDED( hr ) ) 
		{
			if( driverType == D3D_DRIVER_TYPE_HARDWARE ) printf("Created HW Device\n");
			if( driverType == D3D_DRIVER_TYPE_REFERENCE ) printf("Created REFERENCE Device\n");
			if( featureLevel == D3D_FEATURE_LEVEL_11_0 ) printf("Created D3D_FEATURE_LEVEL_11_0 Device\n");
			if( featureLevel == D3D_FEATURE_LEVEL_10_0 ) printf("Created D3D_FEATURE_LEVEL_10_0\n");
			if( featureLevel == D3D_FEATURE_LEVEL_10_1 ) printf("Created D3D_FEATURE_LEVEL_10_1\n");
			break;
		}
	}
}

HRESULT D3D11Object::LoadSamplerState(ID3D11SamplerState** ppSamplerLinear)
{
	printf("Create sampler state\n");

	HRESULT hr = S_OK;
	// Create the sample state
	D3D11_SAMPLER_DESC sampDesc;
	ZeroMemory( &sampDesc, sizeof(D3D11_SAMPLER_DESC) );
	sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	sampDesc.MinLOD = 0;
	sampDesc.MaxLOD = D3D11_FLOAT32_MAX;
	hr = m_pDevice->CreateSamplerState( &sampDesc, ppSamplerLinear );
	return hr;

}

HRESULT D3D11Object::TakeScreenshot()
{
	HRESULT hr = S_OK;
	
	string str;
    LPWSTR str2 = L".\\RayTracerCS\\Screenshots\\test.png";

	cvtLPW2stdstring(str,str2);
	hr = D3DX11SaveTextureToFile(m_pDeviceContext,m_pBackBufferTexture,D3DX11_IFF_PNG,str2);

	if(FAILED(hr))
	{
		printf("Screenshot FAILED\n");
		return hr;
	}
	printf("Screenshot saved at %s\n",str.c_str());
	return hr;
}

HRESULT	D3D11Object::CreateEnvironmentMap( LPCSTR sFileName, ID3D11ShaderResourceView** ppSRV )
{
	HRESULT hr = S_OK;

    ID3D11Resource* pRes = NULL;
    ID3D11Texture2D* pCubeTexture = NULL;
    ID3D11ShaderResourceView* pCubeRV = NULL;

    D3DX11_IMAGE_LOAD_INFO LoadInfo;
    LoadInfo.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE;

	D3DX11CreateTextureFromFileA( m_pDevice, sFileName, &LoadInfo, NULL, &pRes, NULL );
    if( pRes )
    {
		printf("Create environment mapping %s\n", sFileName);
        D3D11_TEXTURE2D_DESC desc;
        ZeroMemory( &desc, sizeof( D3D11_TEXTURE2D_DESC ) );
        pRes->QueryInterface( __uuidof( ID3D11Texture2D ), ( LPVOID* )&pCubeTexture );
        pCubeTexture->GetDesc( &desc );
        SAFE_RELEASE( pRes );

        D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
        ZeroMemory( &SRVDesc, sizeof( SRVDesc ) );
        SRVDesc.Format = desc.Format;
        SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
        SRVDesc.TextureCube.MipLevels = desc.MipLevels;
        SRVDesc.TextureCube.MostDetailedMip = 0;
		hr = m_pDevice->CreateShaderResourceView( pCubeTexture, &SRVDesc, ppSRV );
    }

	return hr;
}

HRESULT D3D11Object::CreateTextureInArray( LPCSTR sFileName, D3DX11_IMAGE_LOAD_INFO* LoadInfo, ID3D11Resource** ppResource, ID3D11Resource* pDest, int iArraySlice )
{
	HRESULT hr = S_OK;

	if(fileExists(sFileName))
	{
		printf("Create texture %s\n", sFileName);
		D3DX11CreateTextureFromFileA( m_pDevice, sFileName, LoadInfo, NULL, (ID3D11Resource**)(ppResource), NULL );
		m_pDeviceContext->CopySubresourceRegion(pDest, D3D11CalcSubresource(0, iArraySlice, 1), 0, 0, 0, *ppResource, 0, NULL);
	}

	return hr;
}

HRESULT D3D11Object::CreateRandomTexture( ID3D11Texture2D* pTexture2D, UINT iWidth, UINT iHeight )
{
	HRESULT hr = S_OK;

	//update random texture
	D3D11_MAPPED_SUBRESOURCE mappedTex;
	m_pDeviceContext->Map(pTexture2D,0,D3D11_MAP_WRITE_DISCARD ,0, &mappedTex );

	for( UINT row = 0; row < iHeight; row++ )
	{
		float* pTexels = (float*)((char*)mappedTex.pData+row * mappedTex.RowPitch);
		for( UINT col = 0; col < iWidth; col++ )
		{
			UINT colStart = col * 4;
			float length;
			do
			{
				pTexels[colStart + 0] = 1.0f-2.0f*float(rand())/float(RAND_MAX);// Red
				pTexels[colStart + 1] = 1.0f-2.0f*float(rand())/float(RAND_MAX);
				pTexels[colStart + 2] = 1.0f-2.0f*float(rand())/float(RAND_MAX);
				pTexels[colStart + 3] = float(rand())/float(RAND_MAX);
				length =	(pTexels[colStart + 0]*pTexels[colStart + 0])+
							(pTexels[colStart + 1]*pTexels[colStart + 1])+
							(pTexels[colStart + 2]*pTexels[colStart + 2]);
			}while(length>1.0f);
			length=sqrt(length);
			pTexels[colStart + 0]/=length;
			pTexels[colStart + 1]/=length;
			pTexels[colStart + 2]/=length;
		}
	}

	m_pDeviceContext->Unmap(pTexture2D,0);
	return hr;
}

#endif