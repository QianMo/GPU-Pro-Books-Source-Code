
#include <Graphics/Dx11/Dx11Renderer.hpp>
#include <Graphics/Camera/Camera.hpp>
#include <Graphics/Camera/Dx11Camera.hpp>
#include <Input/Keyboard.hpp>

#include <d3dcompiler.h>
///<
void Dx11Renderer::UnbindResources(ID3D11DeviceContext* _pContext, const int32 _iIndex, const int32 _iNum)
{
	_pContext->OMSetRenderTargets(0, 0, 0); 
	DVector<ID3D11ShaderResourceView*> nullView(_iNum);	
	_pContext->PSSetShaderResources(_iIndex, _iNum, nullView.Begin());
	_pContext->VSSetShaderResources(_iIndex, _iNum, nullView.Begin());
}

///< Create a 3D texture and Shader Resources.
Texture3D_SV Dx11Renderer::Create3DTexture(ID3D11Device* _pDevice, const DXGI_FORMAT& _format, const Vector3i _dims, const D3D11_SUBRESOURCE_DATA* _pData)
{
	Texture3D_SV rTexture;
	memset(&rTexture,0,sizeof(Texture3D_SV));

	D3D11_TEXTURE3D_DESC DomainDesc;
	memset(&DomainDesc, 0, sizeof(D3D11_TEXTURE3D_DESC));

	DomainDesc.Width  = _dims.x(); 
	DomainDesc.Height = _dims.y();   
	DomainDesc.Depth  = _dims.z();  

	DomainDesc.MipLevels			= 1;
	DomainDesc.Format				= _format;
	DomainDesc.Usage				= D3D11_USAGE_DEFAULT;
	DomainDesc.BindFlags			= D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	DomainDesc.CPUAccessFlags		= 0;

	HRESULT hr = _pDevice->CreateTexture3D(&DomainDesc, _pData, &rTexture._pT);
	ASSERT(hr==S_OK, "3D Texture Failed!");	

	///< Create Shader resources
	if (rTexture._pT)
	{
		D3D11_RESOURCE_DIMENSION type;
		rTexture._pT->GetType(&type);
		ASSERT(type==D3D11_RESOURCE_DIMENSION_TEXTURE3D, "Wrong ressource type!");

		D3D11_TEXTURE3D_DESC Desc;
		rTexture._pT->GetDesc(&Desc);

		D3D11_RENDER_TARGET_VIEW_DESC rtDesc;
		memset(&rtDesc,0,sizeof(D3D11_RENDER_TARGET_VIEW_DESC));
		rtDesc.Format				 = Desc.Format;
		rtDesc.ViewDimension		 = D3D11_RTV_DIMENSION_TEXTURE3D;
		rtDesc.Texture3D.MipSlice	 = 0;
		rtDesc.Texture3D.FirstWSlice = 0;
		rtDesc.Texture3D.WSize		 = _dims.z();

		hr = _pDevice->CreateRenderTargetView(rTexture._pT, &rtDesc, &rTexture._pRTV);
		ASSERT(hr==S_OK, "Failed to create render target view");

		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
		memset(&srvDesc,0,sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));

		srvDesc.Format						= Desc.Format;
		srvDesc.ViewDimension				= D3D11_SRV_DIMENSION_TEXTURE3D;
		srvDesc.Texture3D.MipLevels			= Desc.MipLevels;
		srvDesc.Texture3D.MostDetailedMip	= Desc.MipLevels-1;

		hr = _pDevice->CreateShaderResourceView(rTexture._pT, &srvDesc, &rTexture._pSRV);
		ASSERT(hr==S_OK, "Could not create shader ressource view.  ");
	}
	return rTexture;
}


///<
Texture2D_SV Dx11Renderer::Create2DTexture(ID3D11Device* _pDevice, const DXGI_FORMAT& _format, const Vector2i dims, const D3D11_SUBRESOURCE_DATA* _pData, D3D11_USAGE _usage, uint32 _bind, uint32 _cpuAccess)
{
	Texture2D_SV rTexture;
	memset(&rTexture,0,sizeof(Texture2D_SV));

	ASSERT(_pDevice!=NULL, "Null Device...");	

	D3D11_TEXTURE2D_DESC DomainDesc;
	memset(&DomainDesc, 0, sizeof(D3D11_TEXTURE2D_DESC));

	DomainDesc.Width  = dims.x(); 
	DomainDesc.Height = dims.y();   

	DomainDesc.MipLevels			= 1;
	DomainDesc.ArraySize			= 1;
	DomainDesc.SampleDesc.Count		= 1;
	DomainDesc.SampleDesc.Quality	= 0;

	DomainDesc.Format				= _format;
	DomainDesc.Usage				= _usage;
	DomainDesc.BindFlags			= _bind;
	DomainDesc.CPUAccessFlags		= _cpuAccess;

	HRESULT hr = _pDevice->CreateTexture2D(&DomainDesc, _pData, &rTexture._pT);
	ASSERT(hr==S_OK, "2D Texture Failed!");	
	
	///<
	D3D11_TEXTURE2D_DESC Desc;
	rTexture._pT->GetDesc(&Desc);

	///< SRV
	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	memset(&srvDesc,0,sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));

	srvDesc.Format						= Desc.Format;
	srvDesc.ViewDimension				= D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MipLevels			= Desc.MipLevels;
	srvDesc.Texture2D.MostDetailedMip	= Desc.MipLevels-1;

	hr = _pDevice->CreateShaderResourceView(rTexture._pT, &srvDesc, &rTexture._pSRV);
	ASSERT(hr==S_OK, "Could not create shader ressource view.  ");	

	///< RTV
	if (_bind & D3D11_BIND_RENDER_TARGET)
	{
		D3D11_RENDER_TARGET_VIEW_DESC rtDesc;
		memset(&rtDesc,0,sizeof(D3D11_RENDER_TARGET_VIEW_DESC));
		rtDesc.Format				 = Desc.Format;
		rtDesc.ViewDimension		 = D3D11_RTV_DIMENSION_TEXTURE2D;
		rtDesc.Texture2D.MipSlice	 = 0;		

		hr = _pDevice->CreateRenderTargetView(rTexture._pT, &rtDesc, &rTexture._pRTV);
		ASSERT(hr==S_OK, "Failed to create render target view");
	}


	return rTexture;
}

///<
bool Dx11Renderer::CreateTweakBar()
{
	return MenuManager::Get().Create(m_pDevice, Vector2ui(m_w, m_h));
}

HRESULT Dx11Renderer::CompileShaderFromFile(const char* _strFileName, const char* _strEntryPoint, const char* _strShaderModel, ID3DBlob** _ppBlobOut)
{
	HRESULT hr = S_OK;

	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS | D3D10_SHADER_OPTIMIZATION_LEVEL3;
#if defined( DEBUG ) || defined( _DEBUG )
	// Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
	// Setting this flag improves the shader debugging experience, but still allows 
	// the shaders to be optimized and to run exactly the way they will run in 
	// the release configuration of this program.
	dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

	ID3DBlob* pErrorBlob;
	hr = D3DX11CompileFromFile(_strFileName, NULL, NULL, _strEntryPoint, _strShaderModel, dwShaderFlags, 0, NULL, _ppBlobOut, &pErrorBlob, NULL);
	if (hr!=S_OK)
	{
		if (pErrorBlob != NULL)
			OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
		M::Release(&pErrorBlob);

		return hr;
	}

	M::Release(&pErrorBlob);

	return S_OK;
}



///<
bool Dx11Renderer::CreateDevice(HWND _hWnd)
{
	HRESULT hr = S_OK;

	RECT rc;
	GetClientRect(_hWnd, &rc);
	m_w = rc.right - rc.left;
	m_h = rc.bottom - rc.top;

	UINT createDeviceFlags = 0;
	/*
#if defined(_DEBUG)
	createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
	*/
	D3D_DRIVER_TYPE driverTypes[] =
	{
		D3D_DRIVER_TYPE_HARDWARE,
		D3D_DRIVER_TYPE_WARP,
		D3D_DRIVER_TYPE_REFERENCE,
	};
	UINT numDriverTypes = ARRAYSIZE(driverTypes);

	D3D_FEATURE_LEVEL featureLevels[] =
	{
		//D3D_FEATURE_LEVEL_10_0
		
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0,
		
	};
	UINT numFeatureLevels = ARRAYSIZE(featureLevels);
	//vs_4_0_level_9_1


	DXGI_SWAP_CHAIN_DESC sd;
	memset(&sd, 0, sizeof(sd));
	sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD; 
	sd.BufferCount = 1;
	sd.BufferDesc.Width = m_w;
	sd.BufferDesc.Height = m_h;
	sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.BufferDesc.RefreshRate.Numerator = 60;
	sd.BufferDesc.RefreshRate.Denominator = 1;
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.OutputWindow = _hWnd;
	sd.SampleDesc.Count = 1;
	sd.SampleDesc.Quality = 0;
	sd.Windowed = TRUE;

	D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;

	for (uint32 uiDriverType=0; uiDriverType<numDriverTypes; uiDriverType++)
	{
		hr = D3D11CreateDeviceAndSwapChain(NULL,  driverTypes[uiDriverType], NULL, createDeviceFlags, featureLevels, numFeatureLevels,
			D3D11_SDK_VERSION, &sd, &m_pSwapChain, &m_pDevice, &featureLevel, &m_pImmediateContext);
		if (SUCCEEDED( hr ))
			break;
	}

	if (FAILED(hr))
		return false;

	// Create a render target view
	ID3D11Texture2D* pBackBuffer = NULL;
	hr = m_pSwapChain->GetBuffer(0, __uuidof( ID3D11Texture2D ), ( LPVOID* )&pBackBuffer);
	if (FAILED(hr))
		return false;

	hr = m_pDevice->CreateRenderTargetView(pBackBuffer, NULL, &m_pRenderTargetView);
	pBackBuffer->Release();
	if (FAILED(hr))
		return false;

	///< CreateDepth
	{
		// Create depth stencil texture
		D3D11_TEXTURE2D_DESC descDepth;
		memset(&descDepth, 0, sizeof(descDepth));
		descDepth.Width = m_w;
		descDepth.Height = m_h;
		descDepth.MipLevels = 1;
		descDepth.ArraySize = 1;
		descDepth.Format = DXGI_FORMAT_R32_TYPELESS;//DXGI_FORMAT_D24_UNORM_S8_UINT;
		descDepth.SampleDesc.Count = 1;
		descDepth.SampleDesc.Quality = 0;
		descDepth.Usage = D3D11_USAGE_DEFAULT;
		descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE; 
		descDepth.CPUAccessFlags = 0;
		descDepth.MiscFlags = 0;
		hr = m_pDevice->CreateTexture2D(&descDepth, NULL, &m_pDepthStencilTexture);
		ASSERT(hr==S_OK, "Failed Creating Depth Texture");
		if( FAILED( hr ) )
			return false;

		D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
		memset(&descDSV, 0, sizeof(descDSV));
		descDSV.Format = DXGI_FORMAT_D32_FLOAT; //descDepth.Format;
		descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		descDSV.Texture2D.MipSlice = 0;
		hr = m_pDevice->CreateDepthStencilView(m_pDepthStencilTexture, &descDSV, &m_pDepthStencilView);
		ASSERT(hr==S_OK, "Failed Creating Depth Object");
		if( FAILED( hr ) )
			return false;


		D3D11_SHADER_RESOURCE_VIEW_DESC srDesc;
		srDesc.Format = DXGI_FORMAT_R32_FLOAT;
		srDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		srDesc.Texture2D.MostDetailedMip = 0;
		srDesc.Texture2D.MipLevels = 1;

		hr=m_pDevice->CreateShaderResourceView(m_pDepthStencilTexture, &srDesc, &m_pDepthStencilSRV);
		ASSERT(hr==S_OK, "Failed Creating Depth Object");
		if( FAILED( hr ) )
			return false;
	}

	m_pImmediateContext->OMSetRenderTargets(1, &m_pRenderTargetView, m_pDepthStencilView);

	///< Setup the Viewport
	D3D11_VIEWPORT vp;
	vp.Width	= (float32)m_w;
	vp.Height	= (float32)m_h;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	m_pImmediateContext->RSSetViewports(1, &vp);
	
	List::States::Get().Create(m_pDevice);
		
	CreateTweakBar();

	m_pCamera=new Dx11Camera();
	Camera::Get().Create(m_w,m_h);
	m_pCamera->Create(m_pDevice);
	
	return true;
}

void Dx11Renderer::SetCameraParams()
{
	m_pCamera->SetParams(m_pImmediateContext,0);
}


///<
const char* Dx11Renderer::PSLevel(ID3D11Device* _pDevice)
{ 
	return "ps_4_0";
	/*
	if (_pDevice->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0)
		return "ps_4_0";
	else
		return "ps_5_0";
		*/
}

///<
const char* Dx11Renderer::VSLevel(ID3D11Device* _pDevice)
{
	return "vs_4_0";
	/*
	if (_pDevice->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0)
		return "vs_4_0";
	else
		return "vs_5_0";
		*/
}

///<
const char* Dx11Renderer::GSLevel(ID3D11Device* _pDevice)
{	

	return "gs_4_0";
	/*
	if (_pDevice->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0)
		return "gs_4_0";
	else
		return "gs_5_0";
		*/

}

///<
void Dx11Renderer::CreateGeometryShader(const char* _csFileName, const char* _csGeometryShaderName, ID3D11Device* _pDevice, ID3D11GeometryShader** _ppGeometryShader)
{
	if (_csGeometryShaderName!=NULL)
	{
		ID3DBlob* pGSBlob = NULL;
		HRESULT hr = CompileShaderFromFile(_csFileName, _csGeometryShaderName, GSLevel(_pDevice), &pGSBlob);
		ASSERT(hr==S_OK, "Failed to create shader");		

		if(pGSBlob)
		{	
			hr = _pDevice->CreateGeometryShader(pGSBlob->GetBufferPointer(), pGSBlob->GetBufferSize(), NULL, _ppGeometryShader);
			ASSERT(hr==S_OK, "Failed to create Geometry Shader");
			M::Release(&pGSBlob);	
		}
	}
}

///<
void Dx11Renderer::CreateVertexShader(const char* _csFileName, const char* _csVertexShaderName, ID3D11Device* _pDevice, ID3D11VertexShader** _ppVertexShader)
{
	if (_csVertexShaderName!=NULL)
	{
		ID3DBlob* pVSBlob = NULL;
		HRESULT hr = CompileShaderFromFile(_csFileName, _csVertexShaderName, VSLevel(_pDevice), &pVSBlob);
		ASSERT(hr==S_OK, "Failed to create shader");		

		if(pVSBlob)
		{	
			hr = _pDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), NULL, _ppVertexShader);
			ASSERT(hr==S_OK, "Failed to create Vertex Shader");
			M::Release(&pVSBlob);	
		}	
	}
}

///<
void Dx11Renderer::CreatePixelShader(const char* _csFileName, const char* _csPixelShaderName, ID3D11Device* _pDevice, ID3D11PixelShader** _ppPixelShader)
{
	if (_csPixelShaderName!=NULL)
	{
		ID3DBlob* pPSBlob = NULL;
		HRESULT hr = CompileShaderFromFile(_csFileName, _csPixelShaderName, PSLevel(_pDevice), &pPSBlob);
		ASSERT(hr==S_OK, "Failed Pixel Shader");

		if (pPSBlob)
		{	
			hr = _pDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, _ppPixelShader);
			M::Release(&pPSBlob);
			ASSERT(hr==S_OK, "Failed Pixel Shader");
		}		
	}
}

///<
void Dx11Renderer::CreateInputLayout(const char* _csFileName, const char* _csVertexShaderName, ID3D11Device* _pDevice, D3D11_INPUT_ELEMENT_DESC* _pLayout, int32 _NumElements, ID3D11InputLayout** _ppInputLayout)
{
	if (_csVertexShaderName!=NULL && _pLayout!=NULL)
	{
		ID3DBlob* pVSBlob = NULL;
		HRESULT hr = CompileShaderFromFile(_csFileName, _csVertexShaderName, VSLevel(_pDevice), &pVSBlob);
		ASSERT(hr==S_OK, "Failed to create shader");		

		if(pVSBlob)
		{			
			hr = _pDevice->CreateInputLayout(_pLayout, _NumElements, pVSBlob->GetBufferPointer(),	pVSBlob->GetBufferSize(), _ppInputLayout);
			M::Release(&pVSBlob);
			ASSERT(hr==S_OK, "Failed Layout Creation");

		}
	}
}

///<
void Dx11Renderer::CreateShadersAndLayout(const char* _csFileName, const char* _csVertexShaderName, const char* _csPixelShaderName,  const char* _csGeometryShaderName,
									   D3D11_INPUT_ELEMENT_DESC* _pLayout, int32 _NumElements,
									   Shader* _pShader, ID3D11Device* _pDevice)
{

	ASSERT(_pShader!=NULL, "null pointer!");

	CreateVertexShader(_csFileName,_csVertexShaderName,_pDevice, &_pShader->m_pVertex);	
	CreateInputLayout(_csFileName,_csVertexShaderName,_pDevice,_pLayout,_NumElements,&_pShader->m_pLayout);
	CreatePixelShader(_csFileName,_csPixelShaderName,_pDevice, &_pShader->m_pPixel);
	CreateGeometryShader(_csFileName,_csGeometryShaderName,_pDevice, &_pShader->m_pGeometry);
}


///<
void Dx11Renderer::ReleaseDevice()
{
	List::States::m_inst.Release();	

	M::Delete(&m_pCamera);

	M::Release(&m_pDepthStencilTexture);
	M::Release(&m_pDepthStencilSRV);
	
	M::Release(&m_pDepthStencilView);
	M::Release(&m_pRenderTargetView);

	M::Release(&m_pImmediateContext);
	M::Release(&m_pSwapChain);			
	M::Release(&m_pDevice);	

	TwWindowSize(0, 0);
	TwTerminate();

}



struct VertUV
{
	Vector3f _x;
	Vector2f _uv;
	
};
///<
ID3D11Buffer* Dx11Renderer::CreatePostProcessQuadUVs(ID3D11Device* _pDevice)
{
	VertUV vertices[] =
	{
		{ Vector3f( -1.0f, 1.0f, 0.0f ), Vector2f(0, 0) },

		{ Vector3f( 1.0f, 1.0f, 0.0f ), Vector2f(1, 0) },
		{ Vector3f( 1.0f, -1.0f, 0.0f ), Vector2f(1, 1) },

		{ Vector3f( -1.0f, 1.0f, 0.0f ), Vector2f(0, 0) },
		{ Vector3f( 1.0f, -1.0f, 0.0f ), Vector2f(1, 1) },
		{ Vector3f( -1.0f, -1.0f, 0.0f ), Vector2f(0, 1) }
	};

	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(VertUV)*6;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA InitData;
	memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
	InitData.pSysMem = vertices;

	ID3D11Buffer* pQuadBuffer=NULL;
	HRESULT hr = _pDevice->CreateBuffer(&bd, &InitData, &pQuadBuffer);
	ASSERT(hr==S_OK, "Failed to create Buffer");

	return pQuadBuffer;
}

///<
ID3D11Buffer* Dx11Renderer::CreatePostProcessQuad(ID3D11Device* _pDevice)
{
	Vector3f vertices[] =
	{
		Vector3f( -1.0f, 1.0f, 0.0f ),
		Vector3f( 1.0f, 1.0f, 0.0f ),
		Vector3f( 1.0f, -1.0f, 0.0f ),

		Vector3f( -1.0f, 1.0f, 0.0f ),
		Vector3f( 1.0f, -1.0f, 0.0f ),
		Vector3f( -1.0f, -1.0f, 0.0f )
	};

	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(Vector3f)*6;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA InitData;
	memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
	InitData.pSysMem = vertices;

	ID3D11Buffer* pQuadBuffer=NULL;
	HRESULT hr = _pDevice->CreateBuffer(&bd, &InitData, &pQuadBuffer);
	ASSERT(hr==S_OK, "Failed to create Buffer");

	return pQuadBuffer;
}

