

#include <Apps/GPULiquids/GPU3DParticleLiquids.hpp>

#include <Graphics/Camera/Camera.hpp>
#include <Graphics/Camera/Dx11Camera.hpp>

#include <Graphics/Dx11/Post/FXAA.hpp>

#include <Graphics/Dx11/Utility/Dx11Benchmarks.hpp>
#include <Graphics/Dx11/Shaders.hpp>
#include <Graphics/Dx11/Mesh.hpp>
#include <Graphics\MeshImport.hpp>

#include <Physics/GPUFluids/GPUSPH.hpp>
#include <Physics/GPUFluids/GPUSPHShallowWater.hpp>


///<
bool Simple3DLiquids::Create(HWND _hWnd)
{
	if (CreateDevice(_hWnd))
	{		
		///< Create FXAA
		{
			FXAA::CBAAParams params = {Vector4f(1.0f/m_w, 1.0f/m_h, 0.0f, 0.0f), Vector4f(2,0,0,0)};
			m_pFxaa = new FXAA(params);

			DXGI_SWAP_CHAIN_DESC swapChainDesc;
			m_pSwapChain->GetDesc(&swapChainDesc);
			m_pFxaa->Create(m_pDevice, &swapChainDesc);
		}

		///< Benchmarks
		{
			m_pBench = new TimeStampQuery(m_pDevice);
		}	

		///< GPU Fluid
		{
			const float32 dt=1.0f/30.0f;

			m_pGPUSPH = new GPUSPH();
			m_pGPUSPH->Create(m_pDevice,m_pImmediateContext, Vector3i(64), 40, "WaterHeight.dds");

			if(m_bRayCast)
				CreateRayCast(m_pDevice);

			GPUSPHConstants::Get().CreateContants(m_pDevice, m_pGPUSPH->GetDims());

			CreateContants(m_pDevice);

			Camera::Get().SetDefaultOrbit();
			
			Camera::Get().SetX( Vector3f(0, 24.0f,-72.0f) );//Vector3f(72.0f, 0, 24.0f));
		}

		{			
			MeshImport meshImportTest;
			meshImportTest.Import("..\\..\\Ressources\\Meshes\\SphereNormals.mgf");
			m_pSphere = new Mesh();
			m_pSphere->Create(m_pDevice, &meshImportTest);

			MeshImport meshImportSlope;
			meshImportSlope.Import("..\\..\\Ressources\\Meshes\\Slope.mgf");
			m_pSlope = new Mesh();
			m_pSlope->Create(m_pDevice, &meshImportSlope);
			
			const char* csPhongShaderName = "..\\..\\Src\\Graphics\\Shaders\\Phong.fx";

			m_pPhong = new PhongShader();
			m_pPhong->Create(m_pDevice);

		}
		
		return true;
	}

	return false;
}

///<
void Simple3DLiquids::DrawObstacle(Vector3f _tx, float32 _s)
{
	m_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	m_pPhong->Set(m_pImmediateContext);
	List::States::Get().SetSamplers(m_pImmediateContext);
	m_pImmediateContext->RSSetState(List::States::Get().m_pNoCullBackRasterizer);	

	m_pImmediateContext->OMSetDepthStencilState(List::States::Get().m_pDepthLess, 0);

	Matrix4f translation, scale, rotation;
	M::AffineScale(Vector3f(_s), scale);
	M::AffineTranslation(_tx, translation); 
	m_pSphere->m_World = translation*scale;
	m_pSphere->Draw(m_pImmediateContext);

	{
		Matrix4f rotX, rotY;
		M::AffineScale(Vector3f(7), scale);
		M::AffineTranslation(Vector3f(0,0,0), translation);  
		M::RotationX(-M::Pi*0.5f,rotX);
		M::RotationY(M::Pi*0.5f,rotY);
		m_pSlope->m_World =scale*rotY*rotX;
		M::SetAffineTranslation(Vector3f(-10,0,9.0f), m_pSlope->m_World);

		m_pSlope->Draw(m_pImmediateContext);
	}

	
}


///<
bool Simple3DLiquids::Update()
{
	///< 
	float32 dt=1.0f/30.0f;

	m_pBench->Begin(m_pImmediateContext);
	
	ID3D11RenderTargetView* pCurrentRT = m_pRenderTargetView;
	if (m_pFxaa)
		pCurrentRT = m_pFxaa->GetRenderTarget();	  

	///< Render Scene
	{
		float ClearColor[4] = { 1.0f, 1.0f, 1.0f, 0.0f }; 
		m_pImmediateContext->ClearRenderTargetView(pCurrentRT, ClearColor);
		m_pImmediateContext->ClearDepthStencilView(m_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0 );

		Camera::Get().Update();
		m_pCamera->SetParams(m_pImmediateContext,0);

		{
			GPUSPHConstants::Get().UpdateConstants(m_pImmediateContext);

			m_pGPUSPH->Update(m_pImmediateContext, pCurrentRT, m_pDepthStencilView, Vector2i(m_w,m_h));

			m_pImmediateContext->OMSetBlendState(List::States::Get().m_pDefaultNoBlendState, 0, 0xffffffff);
			List::States::Get().SetSamplers(m_pImmediateContext);
			
			m_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			
			m_pImmediateContext->RSSetState(List::States::Get().m_pNoCullBackRasterizer);	
			m_pImmediateContext->OMSetDepthStencilState(List::States::Get().m_pDepthLess, 0);
			m_pImmediateContext->OMSetRenderTargets(1, &pCurrentRT, m_pDepthStencilView);

			m_pImmediateContext->OMSetRenderTargets(1, &pCurrentRT, m_pDepthStencilView);
			DrawObstacle(10.0f*(Vector3f(0.7f,0.7f,0.2f)*2.0f-Vector3f(1)), 2.5f );

			///< Draw Liquid:
			if (!m_bRayCast)
			{
				m_pImmediateContext->OMSetRenderTargets(1, &pCurrentRT, m_pDepthStencilView);
				m_pGPUSPH->DrawParticles(m_pImmediateContext);
			}	
			else
			{
				ID3D11ShaderResourceView* pVelDensitySRV=m_pGPUSPH->GetDensitySRV();
				m_pRayCastShader->Draw(m_pImmediateContext, pVelDensitySRV, m_pDepthStencilSRV, pCurrentRT, m_pEnvMapTextureSRV);
				m_pImmediateContext->OMSetBlendState(List::States::Get().m_pDefaultNoBlendState, NULL, 0xffffffff);
			}		
			
			
			///< GUI
			m_pImmediateContext->OMSetRenderTargets(1, &pCurrentRT, m_pDepthStencilView);
			TwDraw();
		}		
	}

	/// }
	/// EndPostProcess
	
	if (m_pFxaa)
	{
		m_pImmediateContext->RSSetState(List::States::Get().m_pCullBackRasterizer);
		m_pImmediateContext->OMSetRenderTargets(1, &m_pRenderTargetView, NULL);
		m_pFxaa->Render(m_pImmediateContext);
	}		

	m_fps     = 1.0f/m_pBench->End(m_pImmediateContext);
	
	m_pSwapChain->Present(0, 0);	

	return true;
}

///<
void Simple3DLiquids::Release()
{	
	M::Delete(&m_pGPUSPH);

	M::Delete(&m_pSphere);
	M::Delete(&m_pSlope);
	
	M::Delete(&m_pPhong);

	M::Release(&m_pSliceConstants);
	M::Delete(&m_pBench);

	M::Delete(&m_pQuad);	
	M::Delete(&m_pRawUVSlice);

	M::Delete(&m_pFxaa);

	GPUSPHConstants::Get().Release();
	///< RayCast
	M::Delete(&m_pRayCastShader);	
	M::Release(&m_pEnvMapTextureSRV);
}


///<
void Simple3DLiquids::CreatePostQuad(const Vector2i _iDims)
{
	///< Create Shaders
	{
		{
			const char* csRawUVName = "..\\..\\Src\\Graphics\\Shaders\\Raw.fx"; 

			m_pRawUVSlice = new Shader();

			D3D11_INPUT_ELEMENT_DESC layout[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, sizeof(Vector3f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
			};		

			CreateShadersAndLayout(csRawUVName, "RawVS", "RenderHeightField_PS", NULL, layout, ARRAYSIZE(layout), m_pRawUVSlice, m_pDevice);
		}		

	}


	m_pQuad = new QuadUV();
	m_pQuad->Create(m_pDevice, NULL);
	Matrix4f scale, translation;
	M::AffineScale(Vector3f(9,9,1),scale);
	M::AffineTranslation(Vector3f(0,0,20),translation);
	m_pQuad->m_World = scale*translation;

}

///<
void Simple3DLiquids::CreateRayCast(ID3D11Device* _pDevice)
{
	const char* csRayCastName = "..\\..\\Src\\Graphics\\Shaders\\RayCastLiquid.fx";
	m_pRayCastShader = new RayCastShader();
	Vector3f fdims = m_pGPUSPH->GetDims();
	Vector3f ratios = fdims/fdims.x();
	m_pRayCastShader->Create(csRayCastName,m_pDevice, m_w, m_h);	
	
	
	{
		std::string baseTexture = "..\\..\\Ressources\\Textures\\";

		std::string strEnvMap = std::string("Env.jpg"); 
		if (strEnvMap.length()!=0)
		{
			HRESULT hr = D3DX11CreateShaderResourceViewFromFile(m_pDevice, (baseTexture+strEnvMap).c_str(), NULL, NULL, &m_pEnvMapTextureSRV, NULL );
			ASSERT(hr ==S_OK, "Failed loading texture !");
		}
	}
	
}

///<
void Simple3DLiquids::CreateContants(ID3D11Device* _pDevice)
{
	///<	
	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));

	///< 
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(Constants);
	bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	bd.CPUAccessFlags = 0;
	HRESULT hr = _pDevice->CreateBuffer(&bd, NULL, &m_pSliceConstants);
	ASSERT(hr==S_OK, "Failed To Creat Constant Buffer");	
}


///<
void Simple3DLiquids::CreateMenu()
{
	TwBar* pBar = TwGetBarByIndex(1);

	if (pBar)
	{		
		TwAddVarRO(pBar, "FPS", TW_TYPE_FLOAT, &m_fps, " label='FPS' ");		
		
		GPUSPHConstants::Get().CreateMenu();
		m_pGPUSPH->CreateMenu();
		m_pPhong->CreateMenu();
	}
}
