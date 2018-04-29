

#include <3D/Solver/DX10SolverImpl.hpp>
#include <Common/Math/Random/Uniform.hpp>
#include <Common/Math/Math.hpp>
#include <Common/System/Assert.hpp>

#include <vector>

#ifdef __DX10__


///< Shader from file
#include <d3dX10Async.h>
#include <winuser.h>

#pragma comment( lib, "..\\..\\External\\DirectX\\Lib\\x86\\d3d10.lib" )
#pragma comment( lib, "..\\..\\External\\DirectX\\Lib\\x86\\DXGI.lib" )


#ifdef _DEBUG
#pragma comment( lib, "..\\..\\External\\DirectX\\Lib\\x86\\d3dx10d.lib" )
#else
#pragma comment( lib, "..\\..\\External\\DirectX\\Lib\\x86\\d3dx10.lib" )
#endif

///<
class ScreenSolverState : public SolverState
{
public:
	virtual const char* TechniqueName(){return "Solver";}
	virtual const char* EffectName	(){return "..\\..\\Src\\3D\\Shader\\AppSideSolver10.fx";}
	virtual void		Draw		(DX10RendererImpl* _pRenderer)
	{
		DX10RendererImpl::DrawGPUSolver(_pRenderer);
	}
	virtual void		Create		(DX10RendererImpl* _pRenderer){_pRenderer->CreateGPUSolver();}
		
	virtual float32		GetNumSteps () const {return static_cast<float32>(DX10RendererImpl::NumScreenSteps);}

	virtual ~ScreenSolverState(){}
};

///<
class TDSolverState : public SolverState
{
public:
	virtual const char* TechniqueName(){return "TDSolver";}
	virtual const char* EffectName	(){return "..\\..\\Src\\3D\\Shader\\AppSideSolverTD10.fx";}
	virtual void		Draw		(DX10RendererImpl* _pRenderer)
	{
		DX10RendererImpl::DrawTDGPUSolver(_pRenderer);
		
	}
	virtual void		Create		(DX10RendererImpl* _pRenderer){_pRenderer->CreateTDGPUSolver();}

	virtual float32		GetNumSteps () const {return static_cast<float32>(DX10RendererImpl::NumTDSteps);}

	virtual ~TDSolverState(){}
};

/// 1 Create texture.
/// 2 Create render target view.
/// 3 Create shader resource view.
void LoadTextureAndCreateViews(ID3D10Device* _pDevice, ID3D10Texture2D** _ppTexture, ID3D10ShaderResourceView** _pSRV, 
							   ID3D10RenderTargetView** _pRTV, const DXGI_FORMAT& _format, const char* _strFileName, 
							   D3D10_USAGE _usage = D3D10_USAGE_DEFAULT, UINT _cpuFlag = 0)
{
	ASSERT(_pDevice!=NULL, "Null Device...");
	/// 1 Create texture. 
	D3DX10_IMAGE_LOAD_INFO loadInfo;
	memset(&loadInfo,0, sizeof(D3DX10_IMAGE_LOAD_INFO));

	loadInfo.Usage		= _usage;
	loadInfo.Format		= _format;
	loadInfo.MipLevels	= 1;
	loadInfo.Filter		= D3DX10_FILTER_LINEAR;

	loadInfo.BindFlags |= _pSRV != NULL ? D3D10_BIND_SHADER_RESOURCE : 0;
	loadInfo.BindFlags |= _pRTV != NULL ? D3D10_BIND_RENDER_TARGET : 0;

	loadInfo.CpuAccessFlags=0;
	loadInfo.CpuAccessFlags=_cpuFlag;

	loadInfo.MiscFlags=0;

	HRESULT hr = D3DX10CreateTextureFromFile(_pDevice, _strFileName, &loadInfo, NULL, (ID3D10Resource**)_ppTexture, NULL);
	ASSERT(hr==S_OK, "Could not read texture! ");	

	ID3D10Texture2D* pTexture=*_ppTexture;
	if (pTexture)
	{
		D3D10_RESOURCE_DIMENSION type;
		pTexture->GetType(&type);

		ASSERT(type==D3D10_RESOURCE_DIMENSION_TEXTURE2D, "Wrong ressource type!");

		D3D10_TEXTURE2D_DESC Desc;
		pTexture->GetDesc(&Desc);

		if (_pSRV)
		{
			D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
			srvDesc.Format						= Desc.Format;
			srvDesc.ViewDimension				= D3D10_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MipLevels			= Desc.MipLevels;
			srvDesc.Texture2D.MostDetailedMip	= Desc.MipLevels-1;

			hr = _pDevice->CreateShaderResourceView(pTexture, &srvDesc, _pSRV);
			ASSERT(hr==S_OK, "Could not create shader ressource view.  ");
		}

		if (_pRTV)
		{
			D3D10_RENDER_TARGET_VIEW_DESC rtDesc;
			rtDesc.Format				= Desc.Format;
			rtDesc.ViewDimension		= D3D10_RTV_DIMENSION_TEXTURE2D;
			rtDesc.Texture2D.MipSlice	= 0;

			HRESULT hr = _pDevice->CreateRenderTargetView(pTexture, &rtDesc, _pRTV);
			ASSERT(hr==S_OK, "Failed to create render target view");
		}
	}
}

///<
void DX10RendererImpl::Create(DX10Renderer::SolverType _type)
{	
	if (CreateDevice())
	{
		switch(_type)
		{
			case DX10Renderer::ScreenSolver : m_pState = new ScreenSolverState(); break;
			case DX10Renderer::TDSolver : m_pState = new TDSolverState(); break; 

			default:ASSERT(false, "wrong solver type!");
		}

		m_NumSlices = 32;	

		m_pState->Create(this);
	}
}
///<
void DX10RendererImpl::CreateTDGPUSolver()
{
	D3D10_TEXTURE3D_DESC DomainDesc;
	memset(&DomainDesc, 0, sizeof(D3D10_TEXTURE3D_DESC));


	DomainDesc.Depth  = m_NumSlices;
	DomainDesc.Height = m_NumSlices;
	DomainDesc.Width  = m_NumSlices;

	m_TDSolverBuffer[0].m_FieldDimension = Vector3i(DomainDesc.Width,DomainDesc.Height,DomainDesc.Depth);
	m_TDSolverBuffer[1].m_FieldDimension = m_TDSolverBuffer[0].m_FieldDimension;

	DomainDesc.MipLevels			= 1;
	DomainDesc.Format				= DXGI_FORMAT_R32G32B32A32_FLOAT;
	DomainDesc.Usage				= D3D10_USAGE_DEFAULT;
	DomainDesc.BindFlags			= D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE;
	DomainDesc.CPUAccessFlags = 0;


	Vector4f* pFPixels = new Vector4f[DomainDesc.Height*DomainDesc.Width*DomainDesc.Depth];

	for (uint32 i=0; i<DomainDesc.Depth; ++i)
	{
		for (uint32 j=0; j<DomainDesc.Height; ++j)
		{
			for (uint32 k=0; k<DomainDesc.Width; ++k)
			{
				{
					pFPixels[i*DomainDesc.Height*DomainDesc.Width + j*DomainDesc.Width + k]=Vector4f(0,0,0,1);
				}				
			}
		}
	}

	///< Check fir 3D.
	D3D10_SUBRESOURCE_DATA dbData;
	dbData.pSysMem			= reinterpret_cast<float32*>(pFPixels);
	dbData.SysMemPitch		= DomainDesc.Width*sizeof(Vector4f);
	dbData.SysMemSlicePitch	= DomainDesc.Height*dbData.SysMemPitch;

	HRESULT hr = m_pDevice->CreateTexture3D(&DomainDesc, &dbData, &m_TDSolverBuffer[0].m_pField);
	ASSERT(hr==S_OK, "3D Texture Failed!");
	hr = m_pDevice->CreateTexture3D(&DomainDesc, &dbData, &m_TDSolverBuffer[1].m_pField);
	ASSERT(hr==S_OK, "3D Texture Failed!");

	delete[] pFPixels;

	D3D10_TEXTURE3D_DESC Desc;
	m_TDSolverBuffer[0].m_pField->GetDesc(&Desc);

	D3D10_RENDER_TARGET_VIEW_DESC rtDesc;
	memset(&rtDesc,0,sizeof(D3D10_RENDER_TARGET_VIEW_DESC));
	rtDesc.Format				 = Desc.Format;
	rtDesc.ViewDimension		 = D3D10_RTV_DIMENSION_TEXTURE3D;
	rtDesc.Texture3D.MipSlice	 = 0;
	rtDesc.Texture3D.FirstWSlice = 0;
	rtDesc.Texture3D.WSize = m_NumSlices;

	for (int32 i=0; i<BSize; ++i)
	{
		hr = m_pDevice->CreateRenderTargetView(m_TDSolverBuffer[i].m_pField, &rtDesc, &m_TDSolverBuffer[i].m_pFieldRTV);
		ASSERT(hr==S_OK, "Failed to create render target view");
	}

	D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
	memset(&srvDesc,0,sizeof(D3D10_SHADER_RESOURCE_VIEW_DESC));

	srvDesc.Format						= Desc.Format;
	srvDesc.ViewDimension				= D3D10_SRV_DIMENSION_TEXTURE3D;
	srvDesc.Texture3D.MipLevels			= Desc.MipLevels;
	srvDesc.Texture3D.MostDetailedMip	= Desc.MipLevels-1;

	for (int32 i=0; i<BSize; ++i)
	{
		hr = m_pDevice->CreateShaderResourceView(m_TDSolverBuffer[i].m_pField, &srvDesc, &m_TDSolverBuffer[i].m_pFieldSRV);
		ASSERT(hr==S_OK, "Could not create shader ressource view.  ");
	}
}

///< Load a texture and create a render target view and a shader resource view.
void DX10RendererImpl::CreateGPUSolverBuffer(ID3D10Device* _pDevice, GPUSolver& _GPUSolverBuffer, const char* _strFieldName, const char* _strDensityName)
{
	LoadTextureAndCreateViews(_pDevice, &_GPUSolverBuffer.m_pField, &_GPUSolverBuffer.m_pFieldSRV, &_GPUSolverBuffer.m_pFieldRTV, DXGI_FORMAT_R32G32B32A32_FLOAT, _strFieldName);
	
	D3D10_TEXTURE2D_DESC Desc;
	_GPUSolverBuffer.m_pField->GetDesc(&Desc);
	_GPUSolverBuffer.m_FieldDimension=Vector2i(Desc.Width,Desc.Height);
	
	LoadTextureAndCreateViews(_pDevice, &_GPUSolverBuffer.m_pSmokeDensity, &_GPUSolverBuffer.m_pSmokeDensitySRV, &_GPUSolverBuffer.m_pSmokeDensityRTV, DXGI_FORMAT_R32G32_FLOAT, _strDensityName);

	_GPUSolverBuffer.m_pSmokeDensity->GetDesc(&Desc);
	_GPUSolverBuffer.m_SmokeDensityDimension=Vector2i(Desc.Width,Desc.Height);

}


///<
void DX10RendererImpl::CreateGPUSolver()
{
	std::string strFieldName("..\\..\\Version\\Textures\\InitialDomain.dds");
	std::string strDensityName("..\\..\\Version\\Textures\\InitialDensity.dds");
	
	CreateGPUSolverBuffer(m_pDevice, m_SolverBuffer[0], strFieldName.c_str(), strDensityName.c_str());
	CreateGPUSolverBuffer(m_pDevice, m_SolverBuffer[1], strFieldName.c_str(), strDensityName.c_str());
}

///<
bool DX10RendererImpl::CreateDevice()
{
	HRESULT hr = S_OK;

	IDXGIFactory* pFactory;
	hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)(&pFactory));

	ASSERT(hr==S_OK, "Create Factor Failed! ");

	///< Enumerate Adapters :
	UINT i = 0; 
	IDXGIAdapter* pAdapter; 
	std::vector<IDXGIAdapter*> vAdapters; 
	while (pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND) 
	{ 
		vAdapters.push_back(pAdapter); 
		++i; 
	} 

	UINT	DeviceFlags = 0;
#ifdef _DEBUG
	DeviceFlags |= D3D10_CREATE_DEVICE_DEBUG;
#endif

	D3D10_DRIVER_TYPE DriverTypes[] =
	{
		D3D10_DRIVER_TYPE_HARDWARE,
		D3D10_DRIVER_TYPE_REFERENCE,
	};
	UINT NumDriverTypes = sizeof(DriverTypes)/sizeof(D3D10_DRIVER_TYPE);


	for (UINT driverTypeIndex = 0; driverTypeIndex < NumDriverTypes; driverTypeIndex++)
	{
		m_driverType = DriverTypes[driverTypeIndex];

		hr = D3D10CreateDevice(vAdapters[0],m_driverType, NULL,	DeviceFlags, D3D10_SDK_VERSION,	&m_pDevice);

		if (hr==S_OK)
			break;
	}

	ASSERT(hr==S_OK, "Failed to Create DX10 Device!  ");

	if (m_pDevice)
	{
		DXGI_SWAP_CHAIN_DESC SwapChainDesc;
		ZeroMemory(&SwapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
		SwapChainDesc.BufferCount							= 1;
		SwapChainDesc.BufferDesc.Width						= m_w;
		SwapChainDesc.BufferDesc.Height						= m_h;
		SwapChainDesc.BufferDesc.Format						= DXGI_FORMAT_R8G8B8A8_UNORM;
		SwapChainDesc.BufferDesc.RefreshRate.Numerator		= 60;
		SwapChainDesc.BufferDesc.RefreshRate.Denominator	= 1;
		SwapChainDesc.BufferUsage							= DXGI_USAGE_RENDER_TARGET_OUTPUT;
		SwapChainDesc.OutputWindow							= m_handle;
		SwapChainDesc.SampleDesc.Count						= 1;
		SwapChainDesc.SampleDesc.Quality					= 0;
		SwapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;	
		SwapChainDesc.Windowed								= TRUE;

		hr = pFactory->CreateSwapChain(m_pDevice,&SwapChainDesc,&m_pSwapChain);
		ASSERT(hr==S_OK, "Create Swap Chain Failed!");	

		M::Release(pFactory);
		for(size_t i=0; i<vAdapters.size(); ++i)
			M::Release(&vAdapters[i]);

		if (m_pSwapChain)
		{
			///<
			ID3D10Texture2D* pBuffer;
			hr = m_pSwapChain->GetBuffer(0, __uuidof(ID3D10Texture2D), (LPVOID*)&pBuffer);
			ASSERT(hr==S_OK, "Failed to Get Buffer!  ");

			if (pBuffer)
			{
				hr = m_pDevice->CreateRenderTargetView(pBuffer, NULL, &m_pTRV);
				ASSERT(hr==S_OK, "CreateRenderTargetView ");
				pBuffer->Release();
			}

			D3D10_TEXTURE2D_DESC descDepth;
			descDepth.Width					= m_w;
			descDepth.Height				= m_h;
			descDepth.MipLevels				= 1;
			descDepth.ArraySize				= 1;
			descDepth.Format				= DXGI_FORMAT_D32_FLOAT;

			descDepth.SampleDesc.Count		= 1;
			descDepth.SampleDesc.Quality	= 0;

			descDepth.Usage					= D3D10_USAGE_DEFAULT;
			descDepth.BindFlags				= D3D10_BIND_DEPTH_STENCIL;
			descDepth.CPUAccessFlags		= 0;
			descDepth.MiscFlags				= 0;

			hr = m_pDevice->CreateTexture2D(&descDepth, NULL, &m_pDepthStencil);
			ASSERT(hr==S_OK, "Create Depth Texture failed! ");
			if (m_pDepthStencil)
			{
				D3D10_DEPTH_STENCIL_VIEW_DESC descDSV;

				descDSV.Format				= descDepth.Format;
				descDSV.ViewDimension		= D3D10_DSV_DIMENSION_TEXTURE2D;
				descDSV.Texture2D.MipSlice	= 0;
				hr = m_pDevice->CreateDepthStencilView(m_pDepthStencil, &descDSV, &m_pDSV);
				ASSERT(hr==S_OK, "Create Depth Stencil View Failed  ");
			}			
		}
		else
		{
			M::Release(m_pDevice);
			return false;
		}	
	}

	return true;
}



///<
bool DX10RendererImpl::CreateEffect()
{
	if (!m_pEffect)
	{
		DWORD dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
#if defined(DEBUG) || defined(_DEBUG)
		dwShaderFlags |= D3D10_SHADER_DEBUG;
#endif
		HRESULT hr = D3DX10CreateEffectFromFile(m_pState->EffectName(), NULL, NULL, "fx_4_0", dwShaderFlags, 0, m_pDevice, NULL, NULL, &m_pEffect, NULL, NULL);
		ASSERT(hr==S_OK, "Effect Creation Failed.  ");

		if (m_pEffect)
		{
			m_pTechnique	= m_pEffect->GetTechniqueByName(m_pState->TechniqueName());
			ASSERT(m_pTechnique!=NULL, "technique not found.  ");

			m_pVWorld		= m_pEffect->GetVariableBySemantic("WORLD")->AsMatrix();
			m_pVView		= m_pEffect->GetVariableBySemantic("VIEW")->AsMatrix();
			m_pVViewInverse = m_pEffect->GetVariableBySemantic("VIEWINVERSE")->AsMatrix();
			m_pVProjection	= m_pEffect->GetVariableBySemantic("PROJECTION")->AsMatrix();
		}
		else
		{
			return false;
		}
	}

	return true;

}

///<
bool DX10RendererImpl::CreateScreenGeometry()
{
	if (m_pTechnique)
	{
		if (!m_pVertexLayout)
		{
			D3D10_INPUT_ELEMENT_DESC layout[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, sizeof(D3DXVECTOR4), D3D10_INPUT_PER_VERTEX_DATA, 0 }
			};
			UINT numElements = sizeof(layout)/sizeof(D3D10_INPUT_ELEMENT_DESC);

			D3D10_PASS_DESC PassDesc;
			m_pTechnique->GetPassByIndex(0)->GetDesc(&PassDesc);
			HRESULT hr = m_pDevice->CreateInputLayout(layout, numElements, PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &m_pVertexLayout);
			ASSERT(hr==S_OK, "Screen Quad Input Layout Creation Failed.  ");
			if (!m_pVertexLayout)
				return false;
		}		
		
		if (!m_pVertexBuffer)
		{			
			VertexDefinition* pVertices=new VertexDefinition[m_NumSlices*4];
			for (int32 iz=0; iz<m_NumSlices; ++iz)
			{
				float32 z=static_cast<float32>(iz)/static_cast<float32>(m_NumSlices);
				pVertices[iz*4 + 0]=VertexDefinition( D3DXVECTOR4( -1.0f, 1.0f, 0,  1 ), D3DXVECTOR3(0,0,z) );
				pVertices[iz*4 + 1]=VertexDefinition( D3DXVECTOR4( 1.0f, 1.0f, 0,   1 ), D3DXVECTOR3(1,0,z) );
				pVertices[iz*4 + 2]=VertexDefinition( D3DXVECTOR4( 1.0f, -1.0f, 0,  1 ), D3DXVECTOR3(1,1,z) );
				pVertices[iz*4 + 3]=VertexDefinition( D3DXVECTOR4( -1.0f, -1.0f, 0, 1 ), D3DXVECTOR3(0,1,z) );
			}

			D3D10_BUFFER_DESC bd;
			bd.Usage				= D3D10_USAGE_DEFAULT;
			bd.ByteWidth			= sizeof(VertexDefinition)*m_NumSlices*4;
			bd.BindFlags			= D3D10_BIND_VERTEX_BUFFER;
			bd.CPUAccessFlags		= 0;
			bd.MiscFlags			= 0;

			D3D10_SUBRESOURCE_DATA InitData;
			InitData.pSysMem = pVertices;
			HRESULT hr = m_pDevice->CreateBuffer(&bd, &InitData, &m_pVertexBuffer);
			ASSERT(hr==S_OK, "Failed to create Vertex Buffer !");
			delete[] pVertices;

			if (!m_pVertexBuffer)
				return false;
		}

		if (!m_pIndexBuffer)
		{
			// Create vertex buffer.
			Vector<uint32,6>* pIndices= new Vector<uint32,6>[m_NumSlices];
			
			Vector<uint32,6> base(0,1,2,2,3,0);
			for (int32 d=0; d<m_NumSlices; ++d)
			{
				base+=Vector<uint32,6>(4);
				memcpy(pIndices+d,&base,sizeof(Vector<uint32,6>));				
			}

			D3D10_BUFFER_DESC bd;
			bd.ByteWidth		= sizeof(Vector<uint32,6>)*m_NumSlices;
			bd.Usage			= D3D10_USAGE_DEFAULT;									
			bd.BindFlags		= D3D10_BIND_INDEX_BUFFER;
			bd.CPUAccessFlags	= 0;
			bd.MiscFlags		= 0;

			D3D10_SUBRESOURCE_DATA InitData;
			InitData.pSysMem	= reinterpret_cast<uint32*>(pIndices);
			HRESULT hr = m_pDevice->CreateBuffer(&bd, &InitData, &m_pIndexBuffer);
			ASSERT(hr==S_OK, "Failed to create index buffer!");
			delete[] pIndices;

			if (!m_pIndexBuffer)
				return false;				
		}
		
	}
	
	return true;
}

///<
void DX10RendererImpl::SetScreenSpaceData()
{
	if (m_pVertexLayout)
	{
		m_pDevice->IASetInputLayout(m_pVertexLayout);
	}

	if (m_pVertexBuffer)
	{
		UINT stride = sizeof(VertexDefinition);
		UINT offset = 0;
		m_pDevice->IASetVertexBuffers(0, 1, &m_pVertexBuffer, &stride, &offset);
	}

	if (m_pIndexBuffer)
	{
		m_pDevice->IASetIndexBuffer(m_pIndexBuffer, DXGI_FORMAT_R32_UINT, 0);
	}

	m_pDevice->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

}


///<
void DX10RendererImpl::UpdateFields(DX10RendererImpl* _pRenderer, GPUSolver& _SolverBuffer1, GPUSolver& _SolverBuffer2)
{
	ASSERT(&_SolverBuffer1!=&_SolverBuffer2,"Same Buffer!");

	///< View Port and spaces :
	D3D10_VIEWPORT vp;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;

	///< Update Velocity Field.
	vp.Width	= _SolverBuffer2.m_FieldDimension.x();
	vp.Height	= _SolverBuffer2.m_FieldDimension.y();

	_pRenderer->m_pDevice->RSSetViewports(1, &vp);	
	_pRenderer->m_pDevice->OMSetRenderTargets(1,&_SolverBuffer2.m_pFieldRTV,0);	
	_pRenderer->m_pFieldTexutreVariable->SetResource((ID3D10ShaderResourceView*)_SolverBuffer1.m_pFieldSRV);
	_pRenderer->m_pSmokeDensityTexutreVariable->SetResource((ID3D10ShaderResourceView*)_SolverBuffer1.m_pSmokeDensitySRV);
	_pRenderer->m_pTechnique->GetPassByIndex(0)->Apply(0);	
	_pRenderer->m_pDevice->DrawIndexed(6,0,0);

	///< Decrement refcount.
	_pRenderer->m_pFieldTexutreVariable->SetResource(NULL);

	///< Update Smoke Density Field.
	vp.Width	= _SolverBuffer2.m_SmokeDensityDimension.x();
	vp.Height	= _SolverBuffer2.m_SmokeDensityDimension.y();
	
	_pRenderer->m_pDevice->RSSetViewports(1, &vp);
	_pRenderer->m_pDevice->OMSetRenderTargets(1,&_SolverBuffer2.m_pSmokeDensityRTV,0);	
	_pRenderer->m_pFieldTexutreVariable->SetResource((ID3D10ShaderResourceView*)_SolverBuffer2.m_pFieldSRV);
	_pRenderer->m_pSmokeDensityTexutreVariable->SetResource((ID3D10ShaderResourceView*)_SolverBuffer1.m_pSmokeDensitySRV);
	_pRenderer->m_pTechnique->GetPassByIndex(1)->Apply(0);	
	_pRenderer->m_pDevice->DrawIndexed(6,0,0);

	///< Decrement refcount.
	_pRenderer->m_pFieldTexutreVariable->SetResource(NULL);
	_pRenderer->m_pSmokeDensityTexutreVariable->SetResource(NULL);
}

///<
void DX10RendererImpl::DrawGPUSolver(DX10RendererImpl* _pRenderer)
{
	ASSERT(_pRenderer!=NULL, "NULL Renderer");

	if (!_pRenderer->m_pSmokeDensityTexutreVariable)
	{
		_pRenderer->m_pSmokeDensityTexutreVariable = _pRenderer->m_pEffect->GetVariableBySemantic("DENSITY")->AsShaderResource();
		ID3D10EffectVectorVariable* pSize = _pRenderer->m_pEffect->GetVariableBySemantic("DENSITYSIZE")->AsVector();
		if (pSize)
		{
			Vector2f SDDims=_pRenderer->m_SolverBuffer[0].m_SmokeDensityDimension;
			SDDims=Vector2f(1.0f)/SDDims;
			pSize->SetFloatVector(reinterpret_cast<float32*>(&SDDims));
		}
	}

	if (!_pRenderer->m_pFieldTexutreVariable)
	{
		_pRenderer->m_pFieldTexutreVariable = _pRenderer->m_pEffect->GetVariableBySemantic("FIELD")->AsShaderResource();
		ID3D10EffectVectorVariable* pSize = _pRenderer->m_pEffect->GetVariableBySemantic("FIELDSIZE")->AsVector();
		if (pSize)
		{
			Vector2f FieldDims=_pRenderer->m_SolverBuffer[0].m_FieldDimension;
			FieldDims=Vector2f(1.0f)/FieldDims;
			pSize->SetFloatVector(reinterpret_cast<float32*>(&FieldDims));
		}
	}

	_pRenderer->SetScreenSpaceData();

	for (int32 i=0; i<NumScreenSteps; ++i)
	{
		static int32 Alternate=1;

		///< Update Solver.
		int32 first=Alternate;
		int32 second=(Alternate+1)%2;
		UpdateFields(_pRenderer, _pRenderer->m_SolverBuffer[first], _pRenderer->m_SolverBuffer[second]);
		Alternate=(Alternate+1)%2;
	}

	///< Draw Density.	
	D3D10_VIEWPORT vp;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	vp.Width	= _pRenderer->m_w;
	vp.Height	= _pRenderer->m_h;

	_pRenderer->m_pDevice->RSSetViewports(1, &vp);
	_pRenderer->m_pDevice->OMSetRenderTargets(1,&_pRenderer->m_pTRV,_pRenderer->m_pDSV);
	_pRenderer->m_pSmokeDensityTexutreVariable->SetResource((ID3D10ShaderResourceView*)_pRenderer->m_SolverBuffer[1].m_pSmokeDensitySRV);

	_pRenderer->m_pTechnique->GetPassByIndex(2)->Apply(0);
	_pRenderer->m_pDevice->DrawIndexed(6,0,0);	

	///< Decrement refcount.
	_pRenderer->m_pSmokeDensityTexutreVariable->SetResource(NULL);
}

float32 angle=0.0f;
///<
Vector2f Orbit(float32 _angle, float32 _distance)
{
	return Vector2f(cos(_angle), sin(_angle))*_distance;
}

///<
float32 DX10RendererImpl::Update(const float32 _dt)
{	
	if (CreateEffect())
	{
		if (CreateScreenGeometry())
		{
			if (!m_pQuery)
				m_pQuery=new TimeStampQuery(m_pDevice);

			m_pQuery->Begin();

			m_pSwapChain->Present(0,0);	

			D3DXMATRIX World;
			D3DXMATRIX View;
			D3DXMATRIX ViewInverse;
			D3DXMATRIX Proj;

			D3DXMatrixIdentity(&World);
			D3DXMatrixIdentity(&View);
			D3DXMatrixIdentity(&Proj);

			angle += _dt*0.1f;

			Vector2f XZ = Orbit(angle,1.5f);
			D3DXVECTOR3 Eye = D3DXVECTOR3(XZ.x(), 0.5f, XZ.y());
			D3DXVECTOR3 At = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
			D3DXVECTOR3 Up(0.0f, 1.0f, 0.0f);
			D3DXMatrixLookAtLH(&View, &Eye, &At, &Up);
			D3DXMatrixInverse(&ViewInverse,NULL,&View);

			D3DXMatrixPerspectiveFovLH(&Proj, D3DXToRadian(90), static_cast<FLOAT>(m_w)/static_cast<FLOAT>(m_h), 1.0f, 10.0f );

			m_pVWorld->SetMatrix((float*)&World);
			m_pVView->SetMatrix((float*)&View);
			m_pVViewInverse->SetMatrix((float*)&ViewInverse);
			m_pVProjection->SetMatrix((float*)&Proj);

			ASSERT(m_pTechnique!=NULL,"Null Technique!");

			if (m_bMouseClick)
			{
				m_bMouseClick=false;
				POINT ptCursor;
				GetCursorPos(&ptCursor);
				ScreenToClient(m_handle, &ptCursor);

				Vector3f vPos;
				vPos[0] = Clamp<float32>(ptCursor.x/static_cast<FLOAT>(m_w),0.2f,0.8f);
				vPos[1] = Clamp<float32>(ptCursor.y/static_cast<FLOAT>(m_h),0.2f,0.8f);
				vPos[2] = 0.5f;

				Vector3f vVel(0,0,0);

				if (m_TDSolverBuffer[0].m_pField)
				{
					vVel = Vector3f(0.1f,0.2f,0.5f)*0.07f;

					vPos[1] = vPos[1];
					
				}				

				if (!m_pBlowerPosition)
					m_pBlowerPosition=m_pEffect->GetVariableBySemantic("BLOWER_POSITION")->AsVector();

				if (!m_pBlowerVelocity)
					m_pBlowerVelocity = m_pEffect->GetVariableBySemantic("BLOWER_VELOCITY")->AsVector();

				if (m_pBlowerPosition)
				{
					HRESULT hr = m_pBlowerPosition->SetFloatVector(reinterpret_cast<float32*>(&vPos));
					ASSERT(hr==S_OK, "Failed to set float vector.  ");
				}

				if (m_pBlowerVelocity)
				{
					HRESULT hr = m_pBlowerVelocity->SetFloatVector(reinterpret_cast<float32*>(&vVel));
					ASSERT(hr==S_OK, "Failed to set float vector.  ");
				}
				
			}

			float32 ClearColor[4] = {235.0f, 235.0f, 235.0f, 0.0f}; 
			m_pDevice->ClearRenderTargetView(m_pTRV, ClearColor);
			m_pDevice->ClearDepthStencilView(m_pDSV, D3D10_CLEAR_DEPTH, 1.0f, 0);	

			m_pState->Draw(this);	
			
			float32 fps=m_pQuery->End();
			fps/=m_pState->GetNumSteps();
			

			return fps;
			
		}		
	}

	return 0;
	
}

///<
void DX10RendererImpl::DrawTDGPUSolver(DX10RendererImpl* _pRenderer)
{
	if (!_pRenderer->m_pFieldTexutreVariable)
	{
		_pRenderer->m_pFieldTexutreVariable = _pRenderer->m_pEffect->GetVariableBySemantic("FIELD")->AsShaderResource();
	}

	static int32 src=0;
	static int32 dst=1;

	for (int32 i=0; i<NumTDSteps; ++i)
	{
		_pRenderer->SetScreenSpaceData();

		UpdateTDFields(_pRenderer,_pRenderer->m_TDSolverBuffer[src], _pRenderer->m_TDSolverBuffer[dst]);

		D3D10_VIEWPORT vp;
		vp.MinDepth = 0.0f;
		vp.MaxDepth = 1.0f;
		vp.TopLeftX = 0;
		vp.TopLeftY = 0;
		vp.Width	= _pRenderer->m_w;
		vp.Height	= _pRenderer->m_h;

		_pRenderer->m_pDevice->RSSetViewports(1, &vp);
		_pRenderer->m_pDevice->OMSetRenderTargets(1,&_pRenderer->m_pTRV,_pRenderer->m_pDSV);
		_pRenderer->m_pFieldTexutreVariable->SetResource((ID3D10ShaderResourceView*)_pRenderer->m_TDSolverBuffer[dst].m_pFieldSRV);
		_pRenderer->m_pTechnique->GetPassByIndex(0)->Apply(0);
		_pRenderer->m_particles.Update(_pRenderer);

		std::swap(src,dst);

	}	

	ID3D10Buffer* pBuffers[1];
	pBuffers[0]=_pRenderer->m_particles.m_pDrawToBuffer;
	UINT stride[1] = { sizeof(ParticleVertex) };
	UINT offset[1] = { 0 };
	_pRenderer->m_pDevice->IASetVertexBuffers(0, 1, pBuffers, stride, offset);
	HRESULT hr = _pRenderer->m_particles.m_pHaloTextureVariable->SetResource((ID3D10ShaderResourceView*)_pRenderer->m_particles.m_pHaloSRV);
	ASSERT(hr==S_OK, "Failed to set texture");
	_pRenderer->m_particles.m_pTechnique->GetPassByIndex(1)->Apply(0);
	_pRenderer->m_pDevice->DrawAuto();	

	pBuffers[0]=NULL;
	_pRenderer->m_pDevice->SOSetTargets(1, pBuffers, offset);	
	_pRenderer->m_pFieldTexutreVariable->SetResource(NULL);
	_pRenderer->m_particles.m_pHaloTextureVariable->SetResource(NULL);
	_pRenderer->m_pTechnique->GetPassByIndex(1)->Apply(0);
	
}
///<
void DX10RendererImpl::UpdateTDFields(DX10RendererImpl* _pRenderer, TDSolver& _SolverBuffer1, TDSolver& _SolverBuffer2)
{
	ASSERT(&_SolverBuffer1!=&_SolverBuffer2,"Same Buffer!");

	///< View Port and spaces :
	D3D10_VIEWPORT vp;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;

	///< Update Velocity Field.
	vp.Width	= _SolverBuffer2.m_FieldDimension.x();
	vp.Height	= _SolverBuffer2.m_FieldDimension.y();

	_pRenderer->m_pDevice->RSSetViewports(1, &vp);	
	_pRenderer->m_pDevice->OMSetRenderTargets(1,&_SolverBuffer2.m_pFieldRTV,0);	
	_pRenderer->m_pFieldTexutreVariable->SetResource((ID3D10ShaderResourceView*)_SolverBuffer1.m_pFieldSRV);
	_pRenderer->m_pTechnique->GetPassByIndex(0)->Apply(0);	
	_pRenderer->m_pDevice->DrawIndexed(6*_SolverBuffer2.m_FieldDimension.z(),0,0);

	///< Decrement refcount.
	_pRenderer->m_pFieldTexutreVariable->SetResource(NULL);
	
}
///<
void ParticleSystem::Update(DX10RendererImpl* _pRenderer)
{
	static bool bIsFirst=true;
	if (Create(_pRenderer))
	{
		UINT stride[1] = { sizeof(ParticleVertex) };
		UINT offset[1] = { 0 };

		_pRenderer->m_pDevice->IASetInputLayout(m_pVertexLayout);

		ID3D10Buffer* pBuffers[1];
		pBuffers[0]=m_pDrawFromBuffer;

		_pRenderer->m_pDevice->IASetVertexBuffers(0, 1, pBuffers, stride, offset);
		_pRenderer->m_pDevice->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_POINTLIST);

		pBuffers[0]=m_pDrawToBuffer;

		_pRenderer->m_pDevice->SOSetTargets(1,pBuffers, offset);
		m_pTechnique->GetPassByIndex(0)->Apply(0);
		if (bIsFirst)
			_pRenderer->m_pDevice->Draw(m_NumParticles,0);
		else
			_pRenderer->m_pDevice->DrawAuto();

		pBuffers[0]=NULL;
		_pRenderer->m_pDevice->SOSetTargets(1, pBuffers, offset);

		std::swap(m_pDrawToBuffer,m_pDrawFromBuffer);

	}
}

///<
bool ParticleSystem::Create(DX10RendererImpl* _pRenderer)
{ 
	m_NumParticles=100000;
	if(_pRenderer)
	{
		if (!m_pTechnique)
		{
			m_pTechnique = _pRenderer->m_pEffect->GetTechniqueByName("Particle");
			if (!m_pTechnique)
				return false;
		}

		if (!m_pHalo)
		{
			LoadTextureAndCreateViews(_pRenderer->m_pDevice, &m_pHalo,&m_pHaloSRV,NULL,DXGI_FORMAT_R8G8B8A8_UNORM,"..\\..\\Version\\Textures\\Halo.dds");
			m_pHaloTextureVariable = _pRenderer->m_pEffect->GetVariableBySemantic("HALO")->AsShaderResource();
		}
		
		
		///<
		if (!m_pVertexLayout)
		{			
			D3D10_INPUT_ELEMENT_DESC layout[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 }
			};
			UINT numElements = sizeof(layout)/sizeof(D3D10_INPUT_ELEMENT_DESC);

			D3D10_PASS_DESC PassDesc;
			m_pTechnique->GetPassByIndex(0)->GetDesc(&PassDesc);
			HRESULT hr = _pRenderer->m_pDevice->CreateInputLayout(layout, numElements, PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &m_pVertexLayout);
			ASSERT(hr==S_OK, "Input layout creation failed.  ");
			if (!m_pVertexLayout)
				return false;
		}
		
		///<
		if (!m_pDrawToBuffer && !m_pDrawFromBuffer)
		{
			///< 
			///< Create from texte.
			ID3D10Texture2D* pInitialTexte=0;
			LoadTextureAndCreateViews(_pRenderer->m_pDevice, &pInitialTexte, NULL, NULL, DXGI_FORMAT_R8G8B8A8_UNORM, "..\\..\\Version\\Textures\\InitialDensity.dds", D3D10_USAGE_STAGING, D3D10_CPU_ACCESS_READ);

			D3D10_TEXTURE2D_DESC desc;
			pInitialTexte->GetDesc(&desc);		

			D3D10_MAPPED_TEXTURE2D data;
			HRESULT thr = pInitialTexte->Map(D3D10CalcSubresource(0, 0, 1), D3D10_MAP_READ, 0, &data);
			ASSERT(thr==S_OK, "failed to retrieve texte data");

			std::vector<Vector2f> positions;
			positions.reserve(desc.Height*desc.Width/4);

			Vector2f test = Vector2f(1.0f)/Vector2f(static_cast<float32>(desc.Height), static_cast<float32>(desc.Width));

			Vector4uc* pData=reinterpret_cast<Vector4uc*>(data.pData);
			for (uint32 i=0; i<desc.Height; ++i)
			{
				for(uint32 j=desc.Width-1; j>0; --j)
				{
					if (pData[i*desc.Width + j].y() > 5)
					{
						Vector2f Pos=Vector2f(static_cast<float32>(i),static_cast<float32>(j))/Vector2f(static_cast<float32>(desc.Height), static_cast<float32>(desc.Width));
						positions.push_back(Pos);
					}
				}
			}

			pInitialTexte->Unmap(D3D10CalcSubresource(0, 0, 1));
			pInitialTexte->Release();

			uint32 pIndex=0;
			int32 numSlices=10;
			int32 zIndex=-numSlices/2;
			
			ParticleVertex* pVertices = new ParticleVertex[m_NumParticles];

			bool bRandomInit=false;

			D3DXVECTOR4 Pos;Pos[3]=1;
			for (int32 p=0; p<m_NumParticles; ++p)
			{							
				if (bRandomInit)
				{
					for (int32 i=0;i<3; ++i)
						Pos[i]=(Uniform::Randf()-0.5f)*0.3f;
				}
				else
				{
					Vector2f cPos = (Vector2f(0.5f)-positions[pIndex])*1.0f;
					Pos[0] = cPos.y();
					Pos[1] = cPos.x();
					Pos[2] = (static_cast<float32>(zIndex)/static_cast<float32>(numSlices))*0.15f;

					pIndex = (pIndex+1);
					if(pIndex==positions.size())
					{
						pIndex=0;
						zIndex++;
						if(zIndex>numSlices/2)
							zIndex=-numSlices/2;
					}					
				}

				pVertices[p] = ParticleVertex(Pos);
				
			}

			D3D10_BUFFER_DESC bd =
			{
				m_NumParticles * sizeof(ParticleVertex),
				D3D10_USAGE_DEFAULT,
				D3D10_BIND_VERTEX_BUFFER,
				0,
				0
			};

			bd.BindFlags |= D3D10_BIND_STREAM_OUTPUT | D3D10_BIND_SHADER_RESOURCE;

			D3D10_SUBRESOURCE_DATA InitData;
			ZeroMemory( &InitData, sizeof( D3D10_SUBRESOURCE_DATA ) );
			InitData.pSysMem = pVertices;

			HRESULT hr = _pRenderer->m_pDevice->CreateBuffer(&bd, &InitData, &m_pDrawFromBuffer);
			ASSERT(hr==S_OK, "Failed to create Vertex Buffer!  ");
			hr = _pRenderer->m_pDevice->CreateBuffer(&bd, NULL, &m_pDrawToBuffer);
			ASSERT(hr==S_OK, "Failed to create Vertex Buffer!  ");
			delete[] pVertices;

			if (!m_pDrawToBuffer || !m_pDrawFromBuffer)
				return false;
		}
	}

	return true;

}


///<
void ParticleSystem::Release()
{
	M::Release(&m_pDrawToBuffer);
	M::Release(&m_pDrawFromBuffer);
	M::Release(&m_pIndexBuffer);
	M::Release(&m_pVertexLayout);
	M::Release(&m_pHaloSRV);
	M::Release(&m_pHalo);
}
///<
void GPUSolver::Release()
{
	M::Release(&m_pFieldSRV);
	M::Release(&m_pFieldRTV);

	M::Release(&m_pSmokeDensitySRV);
	M::Release(&m_pSmokeDensityRTV);

	M::Release(&m_pField);
	M::Release(&m_pSmokeDensity);	
}

///<
void TDSolver::Release()
{
	M::Release(&m_pFieldSRV);
	M::Release(&m_pFieldRTV);
	M::Release(&m_pField);
}

///<
void DX10RendererImpl::Release()
{
	M::Delete(&m_pQuery);
	M::Delete(&m_pState);

	m_particles.Release();

	M::Release(&m_pRasterState);
	M::Release(&m_pVertexLayout);
	M::Release(&m_pVertexBuffer);
	M::Release(&m_pIndexBuffer);
	M::Release(&m_pConstants);

	M::Release(&m_pDepthStencil);
	M::Release(&m_pDSV);
	M::Release(&m_pTRV);

	for (int32 i=0; i<BSize; ++i)
	{
		m_TDSolverBuffer[i].Release();
		m_SolverBuffer[i].Release();
	}
	
	M::Release(&m_pEffect);	
	M::Release(&m_pSwapChain);
	M::Release(&m_pDevice);
}

#endif