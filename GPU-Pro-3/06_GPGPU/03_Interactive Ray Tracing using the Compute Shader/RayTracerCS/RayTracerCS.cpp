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

#include "RayTracerCS.h"

#ifdef WINDOWS

RayTracerCS::RayTracerCS(Scene *a_Scene, HWND &m_hWnd)
{								
	printf("Initializing raytracer...\n");

	m_pScene = a_Scene;																										
	m_pMaterial = NULL;

	GRID_SIZE[0] = 64, GRID_SIZE[1] = 64, GRID_SIZE[2] = 1;
	printf("Grid size: %d,%d,%d\n",GRID_SIZE[0],GRID_SIZE[1],GRID_SIZE[2]);

	m_pCamera = new Camera(Point(0.f,0.f,0.f), m_Parser.GetSpeed() );
	m_pCamera->SetCamera(1);
	
	m_pTimeTracker = new Performance();
	m_pDXObject = new D3D11Object();
	m_pDXObject->Initialize( m_hWnd, m_uResult.GetPtrUAV() );

	unsigned int iLightChoice = 1;
	m_pLight = new Light(iLightChoice);
	m_pInput = new Input();

	m_pInput->SetNumBounces( m_Parser.GetNumReflections() );

	Init();

	//Print debug information of the first 5 primitives in the buffer
	//Implement the PrintString function on the generic class, in this case, 'Primitive'
	//DebugCSBuffer<Primitive*>(m_sPrimitives.GetResource(), 5);
}

RayTracerCS::~RayTracerCS(void)
{
	// Shaders
	SAFE_DELETE( m_pMaterial );
	SAFE_DELETE( m_pTimeTracker );
	SAFE_DELETE( m_pInput );
	SAFE_DELETE( m_pCamera );
	SAFE_DELETE( m_pScene );
	SAFE_DELETE( m_pDXObject );
}

void RayTracerCS::Render()
{
	// Update current time
	float fTimer = m_pTimeTracker->updateTime();
	// Reset number of samples if the camera is moved
	if (m_pCamera->Move( fTimer ))
	{
		m_NumMuestras = 0;
	}

	// Update constant buffers
	UpdateCB<cbCamera,Camera>(m_cbCamera.GetResource(), m_pCamera);
	//UpdateCB<cbGlobalIllumination, void>(m_cbGlobalIllumination.GetResource(), NULL);

	// Generate primary rays
	m_csPrimaryRays.Dispatch( m_pDXObject );
	// For each bounce, compute intersections and color
	for(uint32_t r = 0; r < m_pInput->GetNumBounces()+1; r++)
	{
		// Compute Intersections for each reflection
		m_csIntersections.Dispatch( m_pDXObject );
		m_csColor.Dispatch( m_pDXObject );
	}
	
	// Display frame on screen
	m_pDXObject->GetSwapChain()->Present( 0, 0 );
	// Update number of samples
	++m_NumMuestras;
}

HRESULT RayTracerCS::Init()
{
	HRESULT hr = S_OK;

	unsigned int uiNumPrimitives = m_pScene->GetModels()[0]->GetNumPrimitives();
	// Fill Material Buffer for each primitive
	m_pMaterial = new int[uiNumPrimitives];
	for(unsigned int i = 0; i < uiNumPrimitives; i++)
	{
		m_pMaterial[i] = m_pScene->GetModels()[0]->GetPrimitives()[i]->GetMaterial()->GetIndex();
	}

	// Load the acceleration structure buffer
	SelectAccelerationStructure();

	// Load and compile the shaders
	LoadShaders();
	// Load diffuse textures, specular maps, normal maps and environment maps
	LoadTextures();
	

	/*-------------------------------------------------------------
	UAVs and SRVs
	-------------------------------------------------------------*/
	hr = m_uMortonCode.Init(SRV_AND_UAV, STRUCTURED, NULL, sizeof(MortonCode), m_pScene->GetModels()[0]->GetNumPrimitives(), m_pDXObject);

	/*-------------------------------------------------------------
	SRVs
	-------------------------------------------------------------*/
	hr = m_sVertices.Init( SRV, STRUCTURED, &m_pScene->GetModels()[0]->GetVertices()[0], sizeof(Vertex),m_pScene->GetModels()[0]->GetNumVertices(), m_pDXObject );
	hr = m_sIndices.Init( SRV, STRUCTURED, &m_pScene->GetModels()[0]->GetIndices()[0], sizeof(DWORD),m_pScene->GetModels()[0]->GetNumPrimitives()*3, m_pDXObject );
	hr = m_sMaterials.Init( SRV, STRUCTURED, &m_pMaterial[0], sizeof(int),m_pScene->GetModels()[0]->GetNumPrimitives(), m_pDXObject );
	
	if(FAILED(hr))
	{
		return hr;
	}

	m_vpSRViews.push_back(m_sPrimitives.GetSRV());
	m_vpSRViews.push_back(m_sNodes.GetSRV());
	m_vpSRViews.push_back(m_sVertices.GetSRV());
	m_vpSRViews.push_back(m_sIndices.GetSRV());
	m_vpSRViews.push_back(m_sColorTextures.GetSRV());
	m_vpSRViews.push_back(m_sSpecularMapTextures.GetSRV());
	m_vpSRViews.push_back(m_sNormalMapTextures.GetSRV());
	m_vpSRViews.push_back(m_sMaterials.GetSRV());
	m_vpSRViews.push_back(m_sRandomMapTextures.GetSRV());
	m_vpSRViews.push_back(m_sEnvMapTextures.GetSRV());
	//m_vpSRViews.push_back(m_sLBVHNodes.GetSRV());

	ID3D11ShaderResourceView** pSRViews = new ID3D11ShaderResourceView*[m_vpSRViews.size()];
	memcpy( pSRViews, &m_vpSRViews[0], sizeof( pSRViews[0] ) * m_vpSRViews.size() );

	m_pDXObject->GetDeviceContext()->CSSetShaderResources( 0, m_vpSRViews.size(), pSRViews );
	
	/*-------------------------------------------------------------
	UAVs
	-------------------------------------------------------------*/
	// Ray Tracing UAVs
	hr = m_uRays.Init(UAV, STRUCTURED, NULL, sizeof(Ray), WIDTH*HEIGHT, m_pDXObject);
	hr = m_uIntersections.Init(UAV, STRUCTURED, NULL, sizeof(TIntersection)+2*sizeof(int), WIDTH*HEIGHT, m_pDXObject);
	hr = m_uAccumulation.Init(UAV, STRUCTURED, NULL, sizeof(Vector3)+sizeof(float), WIDTH*HEIGHT, m_pDXObject);

	// LBVH Construction
	hr = m_uPrimitives.Init(UAV, STRUCTURED, NULL, sizeof(int), m_pScene->GetModels()[0]->GetNumPrimitives(), m_pDXObject);
	//hr = m_uNodes.Init(UAV, STRUCTURED, NULL, sizeof(LBVHNode), m_pScene->GetModels()[0]->GetAccelStructure()->GetNumberOfElements(), m_pDXObject);

	if(FAILED(hr))
	{
		return hr;
	}
	
	m_vpUAViews.push_back(m_uResult.GetUAV());
	m_vpUAViews.push_back(m_uRays.GetUAV());
	m_vpUAViews.push_back(m_uIntersections.GetUAV());
	m_vpUAViews.push_back(m_uAccumulation.GetUAV());
	m_vpUAViews.push_back(m_uPrimitives.GetUAV());
	m_vpUAViews.push_back(m_uNodes.GetUAV());
	m_vpUAViews.push_back(m_uMortonCode.GetUAV());

	ID3D11UnorderedAccessView** pUAViews = new ID3D11UnorderedAccessView*[m_vpUAViews.size()];
	memcpy( pUAViews, &m_vpUAViews[0], sizeof( pUAViews[0] ) * m_vpUAViews.size() );

	m_pDXObject->GetDeviceContext()->CSSetUnorderedAccessViews( 0, m_vpUAViews.size(), pUAViews, NULL );

	/*-------------------------------------------------------------
	CBs
	-------------------------------------------------------------*/
	hr = m_cbCamera.Init(CONSTANT,STRUCTURED,NULL,sizeof(cbCamera),1,m_pDXObject);
	hr = m_cbLight.Init(CONSTANT,STRUCTURED,NULL,sizeof(cbLight),1,m_pDXObject);
	hr = m_cbUserInput.Init(CONSTANT,STRUCTURED,NULL,sizeof(cbInputOutput),1,m_pDXObject);
	hr = m_cbGlobalIllumination.Init(CONSTANT,STRUCTURED,NULL,sizeof(cbGlobalIllumination),1,m_pDXObject);

	if(FAILED(hr))
	{
		return hr;
	}
	
	m_vpCBuffers.push_back((ID3D11Buffer*)m_cbCamera.GetResource());
	m_vpCBuffers.push_back((ID3D11Buffer*)m_cbUserInput.GetResource());
	m_vpCBuffers.push_back((ID3D11Buffer*)m_cbLight.GetResource());
	m_vpCBuffers.push_back((ID3D11Buffer*)m_cbGlobalIllumination.GetResource());

	ID3D11Buffer** pCBuffers = new ID3D11Buffer*[m_vpCBuffers.size()];
	memcpy( pCBuffers, &m_vpCBuffers[0], sizeof( pCBuffers[0] ) * m_vpCBuffers.size() );

	m_pDXObject->GetDeviceContext()->CSSetConstantBuffers(0, m_vpCBuffers.size(), pCBuffers);

	// Send data to constant buffers
	UpdateCB<cbCamera,Camera>(m_cbCamera.GetResource(), m_pCamera);
	UpdateCB<cbInputOutput,Input>(m_cbUserInput.GetResource(), m_pInput);
	UpdateCB<cbLight,Light>(m_cbLight.GetResource(), m_pLight);
	UpdateCB<cbGlobalIllumination, void>(m_cbGlobalIllumination.GetResource(), NULL);

	return hr;
}

//---------------------------------------------
// Loads geometry texture.
//---------------------------------------------
HRESULT RayTracerCS::LoadTextures()
{
	HRESULT hr = S_OK;
	// create materials buffer
	Material** materials = m_pScene->GetModels()[0]->GetMaterials();
	unsigned int numMaterials = m_pScene->GetModels()[0]->GetNumMaterials();

	// Create buffers for textures
	hr = m_sColorTextures.Init(SRV, TEXTURE2D, NULL, 0, numMaterials, m_pDXObject);
	if(FAILED(hr))
	{
		printf("FAILED to init m_sColorTextures.\n");
		return hr;
	}
	
	hr = m_sSpecularMapTextures.Init(SRV, TEXTURE2D, NULL, 0, numMaterials, m_pDXObject);
	if(FAILED(hr))
	{
		printf("FAILED to init m_sSpecularMapTextures.\n");
		return hr;
	}
	
	hr = m_sNormalMapTextures.Init(SRV, TEXTURE2D, NULL, 0, numMaterials, m_pDXObject);
	if(FAILED(hr))
	{
		printf("FAILED to init m_sNormalMapTextures.\n");
		return hr;
	}
	
	// create temporary textures
	ID3D11Texture2D* pTextureColor2D;
	ID3D11Texture2D* pTextureNormalMap2D;
	ID3D11Texture2D* pTextureSpecularMap2D;

	D3D11_TEXTURE2D_DESC dstex;
	ZeroMemory( &dstex, sizeof(dstex) );
    dstex.Width = 1024;
    dstex.Height = 1024;
    dstex.MipLevels = 1;
	dstex.ArraySize = numMaterials;
    dstex.SampleDesc.Count = 1;
    dstex.SampleDesc.Quality = 0;
    dstex.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dstex.Usage = D3D11_USAGE_DEFAULT;
    dstex.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    dstex.CPUAccessFlags = 0;
    dstex.MiscFlags = 0;
	
    hr = m_pDXObject->GetDevice()->CreateTexture2D( &dstex, NULL, &pTextureColor2D );
	if(FAILED(hr))
	{
		printf("FAILED to create TextureColor2D.\n");
		return hr;
	}
	hr = m_pDXObject->GetDevice()->CreateTexture2D( &dstex, NULL, &pTextureNormalMap2D );
	if(FAILED(hr))
	{
		printf("FAILED to create pTextureNormalMap2D.\n");
		return hr;
	}
	hr = m_pDXObject->GetDevice()->CreateTexture2D( &dstex, NULL, &pTextureSpecularMap2D );
	if(FAILED(hr))
	{
		printf("FAILED to create pTextureSpecularMap2D.\n");
		return hr;
	}

	// Load Textures
	D3DX11_IMAGE_LOAD_INFO loadInfo;
	ZeroMemory( &loadInfo, sizeof(D3DX11_IMAGE_LOAD_INFO) );
	loadInfo.Width = 1024;
	loadInfo.Height = 1024;
	loadInfo.Depth = 1;
	loadInfo.FirstMipLevel = 0;
	loadInfo.MipLevels = 1;
	loadInfo.Usage = D3D11_USAGE_DEFAULT;
	loadInfo.BindFlags = 0;
	loadInfo.CpuAccessFlags = 0;
	loadInfo.MiscFlags = 0;
	loadInfo.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	loadInfo.Filter = D3DX11_DEFAULT;
	loadInfo.MipFilter = D3DX11_DEFAULT;
	loadInfo.pSrcInfo = NULL;    

	const string path = "./Models/Textures/";
	for(unsigned int i = 0; i < numMaterials; ++i)
	{	
		string name = "";
		string textureName = m_pScene->GetModels()[0]->GetMaterials()[i]->GetTextureName();
		if(textureName.empty()) 
			continue;

		// Load diffuse texture
		name = path + textureName;
		hr = m_pDXObject->CreateTextureInArray(name.c_str(), &loadInfo, (ID3D11Resource**)(&pTextureColor2D), m_sColorTextures.GetResource(), i);
		if(FAILED(hr))
		{
			printf("FAILED to create pTextureNormalMap2D.\n");
			return hr;
		}

		// Load specular map
		textureName = m_pScene->GetModels()[0]->GetMaterials()[i]->GetSpecularMap();
		name = path + textureName;
		hr = m_pDXObject->CreateTextureInArray(name.c_str(), &loadInfo, (ID3D11Resource**)(&pTextureSpecularMap2D), m_sSpecularMapTextures.GetResource(), i);
		if(FAILED(hr))
		{
			printf("FAILED to create pTextureSpecularMap2D.\n");
			return hr;
		}

		// Load normal map
		textureName = m_pScene->GetModels()[0]->GetMaterials()[i]->GetNormalMap();
		name = path + textureName;
		hr = m_pDXObject->CreateTextureInArray(name.c_str(), &loadInfo,(ID3D11Resource**)(&pTextureNormalMap2D), m_sNormalMapTextures.GetResource(), i);
		if(FAILED(hr))
		{
			printf("FAILED to create pTextureNormalMap2D.\n");
			return hr;
		}
	}

	// Create Random Texture
	dstex.Usage = D3D11_USAGE_DYNAMIC;
	dstex.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	dstex.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	dstex.ArraySize = 1;
	dstex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	hr = m_pDXObject->GetDevice()->CreateTexture2D( &dstex, NULL, (ID3D11Texture2D**)m_sRandomMapTextures.GetPtrResource() );

	if(FAILED(hr))
	{
		printf("FAILED to create m_sRandomMapTextures.\n");
		return hr;
	}

	hr = m_pDXObject->CreateRandomTexture( (ID3D11Texture2D*)m_sRandomMapTextures.GetResource(), dstex.Width, dstex.Height );

	if(FAILED(hr))
	{
		printf("FAILED to fill m_sRandomMapTextures.\n");
		return hr;
	}

	// Load Random Texture
	D3D11_SHADER_RESOURCE_VIEW_DESC desc;
    ZeroMemory( &desc, sizeof(desc) );
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.Texture2DArray.MostDetailedMip = 0;
	desc.Texture2DArray.MipLevels = 1;
	desc.Texture2DArray.FirstArraySlice = 0;
	desc.Texture2DArray.ArraySize = numMaterials;
	desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	desc.Texture2DArray.ArraySize = 1;
	hr = m_pDXObject->GetDevice()->CreateShaderResourceView( (ID3D11Resource*)m_sRandomMapTextures.GetResource(), &desc, m_sRandomMapTextures.GetPtrSRV() );

	if(FAILED(hr))
	{
		printf("FAILED to create resource for m_sRandomMapTextures.\n");
		return hr;
	}

	// Load Environment Mapping
	string name = path+"rnl_cross.dds";
	hr = m_pDXObject->CreateEnvironmentMap(name.c_str(), m_sEnvMapTextures.GetPtrSRV());
	if(FAILED(hr))
	{
		printf("FAILED to create m_sEnvMapTextures.\n");
		return hr;
	}

	return hr;
}


void RayTracerCS::SelectAccelerationStructure()
{
	printf("Acceleration structure selected: ");
	if(m_pScene->GetModels()[0]->GetAccelStructure()->GetName() == "BVH")
	{
		printf("BVH\n");	
		BVH* bvh = (BVH*)m_pScene->GetModels()[0]->GetAccelStructure();	

		int* m_vPrimitives = new int[m_pScene->GetModels()[0]->GetNumPrimitives()];
		for (unsigned int i = 0; i < m_pScene->GetModels()[0]->GetNumPrimitives(); ++i ) 
		{
			m_vPrimitives[i] = bvh->GetPrimitives()[i].primitiveNumber;
		}
			
		m_sNodes.Init(SRV, STRUCTURED, &bvh->GetNodes()[0], sizeof(LinearBVHNode), bvh->GetNumberOfElements(), m_pDXObject);
		m_sPrimitives.Init(SRV, STRUCTURED, &m_vPrimitives[0], sizeof(int), m_pScene->GetModels()[0]->GetNumPrimitives(), m_pDXObject);

		m_pInput->SetAccelerationStructureFlag(0);
	}
	/*else if(m_pScene->GetModels()[0]->GetAccelStructure()->GetName() == "LBVH")
	{
		printf("LBVH\n");	
		LBVH* lbvh = (LBVH*)m_pScene->GetModels()[0]->GetAccelStructure();	
		m_sLBVHNodes.Init(SRV, STRUCTURED, &lbvh->getNodes()[0], sizeof(LBVHNode), lbvh->GetNumberOfElements(), m_pDXObject);
		m_sPrimitives.Init(SRV, STRUCTURED, &lbvh->getPrimitives()[0], sizeof(int), m_pScene->GetModels()[0]->GetNumPrimitives(), m_pDXObject);

		m_pInput->SetAccelerationStructureFlag(9);
	}*/
	else
	{
		printf("ERROR-> Wrong acceleration structure input.\n");	
		PostQuitMessage(0);
	}
}

//---------------------------------------------
// Load/Compile Vertex/Pixel/CS Shaders
//---------------------------------------------
void RayTracerCS::LoadShaders()
{
	// Compile and create compute shader
	unsigned int N = 0;
	std::vector<std::pair<string, int>> macros(4);
	macros[0] = std::pair<string,int>("BLOCK_SIZE_X",GROUP_SIZE[0]);
	macros[1] = std::pair<string,int>("BLOCK_SIZE_Y",GROUP_SIZE[1]);
	macros[2] = std::pair<string,int>("BLOCK_SIZE_Z",GROUP_SIZE[2]);
	macros[3] = std::pair<string,int>("N",N);

	GRID_SIZE[0] = 64;
	GRID_SIZE[1] = 64;
	GRID_SIZE[2] = 1;

	GROUP_SIZE[0] = 16;
	GROUP_SIZE[1] = 16;
	GROUP_SIZE[2] = 1;

	macros[0].second = GROUP_SIZE[0];
	macros[1].second = GROUP_SIZE[1];
	macros[2].second = GROUP_SIZE[2] ;
	macros[3].second = GROUP_SIZE[0] * GRID_SIZE[0];

	m_csPrimaryRays.Load((WCHAR*)L"./RayTracerCS/Shaders/App.hlsl",(LPCSTR)"CSGeneratePrimaryRays", m_pDXObject, macros);
	m_csPrimaryRays.SetDimensiones(GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2]);
	m_csIntersections.Load((WCHAR*)L"./RayTracerCS/Shaders/App.hlsl",(LPCSTR)"CSComputeIntersections", m_pDXObject, macros);
	m_csIntersections.SetDimensiones(GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2]);
	m_csColor.Load((WCHAR*)L"./RayTracerCS/Shaders/App.hlsl",(LPCSTR)"CSComputeColor",m_pDXObject, macros);
	m_csColor.SetDimensiones(GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2]);
}

/*------------------------------------------------------------------------------------------------------
This function updates a constant buffer by simulating herency on the structures defined
at ConstantBuffers.h
------------------------------------------------------------------------------------------------------*/
template <class R, class T>
HRESULT RayTracerCS::UpdateCB(ID3D11Resource* pResource, T* pObj)
{
	HRESULT hr;

	D3D11_MAPPED_SUBRESOURCE mp; 
	hr = m_pDXObject->GetDeviceContext()->Map(pResource,0,D3D11_MAP_WRITE_DISCARD,0,&mp);
	if(FAILED(hr))
	{
		printf("FAILED updating buffer.\n");
		return hr;
	}
	R* tmp = (R*)mp.pData;
	tmp->Update( pObj );
	m_pDXObject->GetDeviceContext()->Unmap(pResource,0);

	return S_OK;
}

/*------------------------------------------------------------------------------------------------------
It prints on the output window the content of the buffer. It is useful for debugging 
purposes. As far as we know, this is the only option for debugging the CS buffers.
------------------------------------------------------------------------------------------------------*/
template <class U>
HRESULT RayTracerCS::DebugCSBuffer ( ID3D11Resource* pBuffer, int iEnd, int iStart ) 
{
	HRESULT hr = S_OK;
	ID3D11Buffer* debugBuf = m_pDXObject->CreateAndCopyToDebugBuf( static_cast<ID3D11Buffer*>(pBuffer) );
    D3D11_MAPPED_SUBRESOURCE MappedResource; 
    hr = m_pDXObject->GetDeviceContext()->Map( debugBuf, 0, D3D11_MAP_READ, 0, &MappedResource );
	if(FAILED(hr))
	{
		printf("FAILED debug function on mapping resources.\n");
		return hr;
	}
	U* p = (U*)MappedResource.pData;

	printf("\n-----------------------------------\nDEBUG %s\n-----------------------------------\n", typeid(U).name());
	for(int i = iStart; i < iEnd; ++i)
	{
		printf("{%d}\n",i);
		p[i]->PrintString();
		printf("\n");
	}

    m_pDXObject->GetDeviceContext()->Unmap( debugBuf, 0 );
	debugBuf->Release();	

	return hr;
}

/*------------------------------------------------------------------------------------------------------
It handles the user input actions
------------------------------------------------------------------------------------------------------*/
LRESULT CALLBACK RayTracerCS::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_LBUTTONDOWN:	m_pInput->SetMouseDown(true);	break;
	case WM_LBUTTONUP:		m_pInput->SetMouseDown(false);	break;
	case WM_MOUSEMOVE:
		if( m_pInput->IsMouseDown() )
		{
			UINT oldMouseX;
			UINT oldMouseY;
			m_pInput->OnMouseMove( oldMouseX, oldMouseY, lParam );
			m_pCamera->Rotate( oldMouseX,oldMouseY );
			m_NumMuestras = 0;
		}
		m_pInput->SetMouseCoordinates(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch ( wParam )
		{
		case 'W':			m_pCamera->SetFront(true); 		break;
		case 'A':			m_pCamera->SetLeft(true);		break;
		case 'S':			m_pCamera->SetBack(true);		break;
		case 'D':			m_pCamera->SetRight(true);		break;
		case 'Q':			m_pCamera->SetDown(true);		break;
		case 'E':			m_pCamera->SetUp(true);			break;
		case VK_ESCAPE:		printf("QUIT\n"); PostQuitMessage(0);				break;
		case VK_LEFT:		if(GetKeyState(VK_CONTROL) < 0){}
							else m_pCamera->Turn(1,-22.5); 
							break;
		case VK_RIGHT:		if(GetKeyState(VK_CONTROL) < 0){}
							else m_pCamera->Turn(1,22.5);
							break;
		case VK_DOWN:		if(GetKeyState(VK_CONTROL) < 0) m_pCamera->DecreaseSpeed();
							else  m_pCamera->Turn(0,-22.5);
							break;
		case VK_UP:			if(GetKeyState(VK_CONTROL) < 0) m_pCamera->IncreaseSpeed();
							else m_pCamera->Turn(0,22.5); 
							break;
		case VK_HOME:		m_pCamera->Turn(2,22.5);
							break;
		case VK_END :		m_pCamera->Turn(2,-22.5);
							break;
		case 'P':			m_pCamera->ChangePausingState(); 
							UpdateCB<cbInputOutput,Input>(m_cbUserInput.GetResource(), m_pInput); 
							break;
		case 'F':			m_pInput->ChangePhongShadingState(); 
							UpdateCB<cbInputOutput,Input>(m_cbUserInput.GetResource(), m_pInput); 
							break;
		case 'K':			m_pInput->ChangeShadowingState(); 
							UpdateCB<cbInputOutput,Input>(m_cbUserInput.GetResource(), m_pInput); 
							break;
		case 'N':			m_pInput->ChangeNormalMappingState(); 
							UpdateCB<cbInputOutput,Input>(m_cbUserInput.GetResource(), m_pInput); 
							break;
		case 'M':			m_pInput->ChangeEnvMappingFlag(); 
							UpdateCB<cbInputOutput,Input>(m_cbUserInput.GetResource(), m_pInput); 
							break;
		case 'G':			m_pInput->ChangeGlossMappingState(); 
							UpdateCB<cbInputOutput,Input>(m_cbUserInput.GetResource(), m_pInput); 
							break;
		case VK_SPACE:		m_pDXObject->TakeScreenshot(); 
							break;
		case VK_NUMPAD0:	
							if(GetKeyState(VK_CONTROL) < 0) 
							{
								m_pLight->SelectLight(0);
								UpdateCB<cbLight,Light>(m_cbLight.GetResource(), m_pLight);
							} 
							else	{ m_pCamera->SetCamera(0); }
							break;
		case VK_NUMPAD1:	
							if(GetKeyState(VK_CONTROL) < 0) 
							{ 
								m_pLight->SelectLight(1);
								UpdateCB<cbLight,Light>(m_cbLight.GetResource(), m_pLight);
							} 
							else	{ m_pCamera->SetCamera(1); }
							break;
		case VK_NUMPAD2:	m_pCamera->SetCamera(2); break; /*sponza @lion*/
		case VK_NUMPAD3:	m_pCamera->SetCamera(3); break;
		case VK_NUMPAD4:	m_pCamera->SetCamera(4); break;
		case VK_NUMPAD5:	m_pCamera->SetCamera(5); break;
		case VK_NUMPAD6:	m_pCamera->SetCamera(6); break;
		case VK_NUMPAD7:	m_pCamera->SetCamera(7); break;
		case VK_NUMPAD8:	m_pCamera->SetCamera(8); break;
		case VK_NUMPAD9:	m_pCamera->SetCamera(9); break;
		case VK_F4:			m_pCamera->ChangeOrbitingState(); break;
		}
		break;
	case WM_KEYUP:
		switch ( wParam )
		{
			case '0':	m_pScene->ChangeStructure(AS_BVH); 	Init(); break;
			case '9':	m_pScene->ChangeStructure(AS_LBVH); Init(); break;
			case 'W':	m_pCamera->SetFront(false);	break;
			case 'A':	m_pCamera->SetLeft(false);	break;
			case 'S':	m_pCamera->SetBack(false);	break;
			case 'D':	m_pCamera->SetRight(false);	break;
			case 'Q':	m_pCamera->SetDown(false);	break;
			case 'E':	m_pCamera->SetUp(false); break;
			case 'L':
				// Shaders
				m_csPrimaryRays.Release();
				m_csIntersections.Release();
				m_csColor.Release();
				LoadShaders();
				m_NumMuestras=0;
				break;
			case VK_ADD:		
				m_pInput->OperateOnBounces(+1);
				UpdateCB<cbInputOutput,Input>(m_cbUserInput.GetResource(), m_pInput);		
				break;
			case VK_SUBTRACT:	
				if (m_pInput->GetNumBounces() > 0)
					m_pInput->OperateOnBounces(-1);
				UpdateCB<cbInputOutput,Input>(m_cbUserInput.GetResource(), m_pInput);
				break;
		}
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
		break;
	}
	return 0;
}

#endif