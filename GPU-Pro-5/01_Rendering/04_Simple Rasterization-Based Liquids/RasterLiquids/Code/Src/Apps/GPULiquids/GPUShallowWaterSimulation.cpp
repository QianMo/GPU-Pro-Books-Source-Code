

#include <Apps/GPULiquids/GPUShallowWaterSimulation.hpp>

#include <Graphics/Camera/Camera.hpp>
#include <Graphics/Camera/Dx11Camera.hpp>
#include <Graphics/Dx11/Post/FXAA.hpp>
#include <Graphics/Dx11/Utility/Dx11Benchmarks.hpp>
#include <Graphics/Dx11/Shaders.hpp>
#include <Graphics/Dx11/Mesh.hpp>

#include <Physics/GPUFluids/GPUSPH.hpp>
#include <Physics/GPUFluids/GPUSPHShallowWater.hpp>
#include <Physics/Terrain/Terrain.hpp>

///<
void ParticleShallowWater::CreateSurfaces()
{
	{
		m_pWaterSurface=new Terrain();
		
		m_pWaterSurface->Create(m_pDevice, m_pShallowWater->GetDims());

		D3D11_INPUT_ELEMENT_DESC layoutDisplacement[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, sizeof(Vector3f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
		};

		const char* csWaterSurfaceShaderName = "..\\..\\Src\\Graphics\\Shaders\\Physics\\WaterSurfaceHF.fx";
		m_pWaterSurfaceShader = new Shader();
		Dx11Renderer::CreateShadersAndLayout(csWaterSurfaceShaderName, "PhongDisplacementVS", "PhongPS", NULL, layoutDisplacement, ARRAYSIZE(layoutDisplacement), m_pWaterSurfaceShader, m_pDevice);

		m_pPhong=new PhongShader();
		m_pPhong->Create(m_pDevice);

	}

	{
		m_pTerrain=new Terrain();
		m_pTerrain->CreateWithHeightTexture(m_pDevice, "BottomAndMountains.png");

	}
	
	Matrix4f translation,scale;

	M::AffineScale(Vector3f(1),scale);

	Vector3f baseX=Vector3f(0,0,0);
	M::AffineTranslation(baseX,translation);

	m_pWaterSurface->m_World = scale*translation;
	m_pTerrain->m_World = scale*translation;


}

///<
void ParticleShallowWater::CreatePostQuad()
{
	///< Create Shaders
	{
		{
			const char* csRawUVName = "..\\..\\Src\\Graphics\\Shaders\\Raw.fx";

			m_pRawUV = new Shader();

			// Define the input layout
			D3D11_INPUT_ELEMENT_DESC layout[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, sizeof(Vector3f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
			};		

			CreateShadersAndLayout(csRawUVName, "RawVS", "RenderHeightField_PS", NULL, layout, ARRAYSIZE(layout), m_pRawUV, m_pDevice);
		}		

	}


	m_pQuad = new QuadUV();
	m_pQuad->Create(m_pDevice, NULL);
	Matrix4f scale, translation;
	M::AffineScale(Vector3f(9,9,1),scale);
	M::AffineTranslation(Vector3f(0,0,60),translation);
	m_pQuad->m_World = scale*translation;

}

///<
bool ParticleShallowWater::Create(HWND _hWnd)
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
			m_pShallowWater = new GPUSPHShallowWater();
			m_pShallowWater->Create(m_pDevice, m_pImmediateContext, "OceanBottom.png");
			Vector2i shDims = m_pShallowWater->GetDims();

			GPUSPHConstants::Get().CreateContants(m_pDevice, Vector3i(shDims.x(),shDims.y(),0));
			m_pShallowWater->ModifyConstants();
		}

		CreatePostQuad();

		CreateSurfaces();

		Camera::Get().SetDefaultOrbit();
		Camera::Get().SetX( 6.0f*Vector3f(0.0f, 24.0f, 72.0f));


		return true;
	}
	return false;
}

///<
bool ParticleShallowWater::Update()
{
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

		GPUSPHConstants::Get().UpdateConstants(m_pImmediateContext);
		m_pShallowWater->Update(m_pImmediateContext,pCurrentRT,m_pDepthStencilView, Vector2i(m_w,m_h));

		m_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		m_pImmediateContext->OMSetBlendState(List::States::Get().m_pDefaultNoBlendState,  0, 0xffffffff);
		m_pImmediateContext->RSSetState(List::States::Get().m_pNoCullBackRasterizer);	
		m_pImmediateContext->OMSetDepthStencilState(List::States::Get().m_pDepthLess, 0);

		m_pImmediateContext->OMSetRenderTargets(1, &pCurrentRT, m_pDepthStencilView);

		D3D11_VIEWPORT vp;
		vp.MinDepth = 0.0f;   vp.MaxDepth = 1.0f;  vp.TopLeftX = 0;   vp.TopLeftY = 0;
		vp.Width	= (float32)m_w;    		vp.Height	= (float32)m_h;
		m_pImmediateContext->RSSetViewports(1, &vp);

		///< Draw Surface:
		{
			m_pWaterSurfaceShader->Set(m_pImmediateContext);
			ID3D11ShaderResourceView* pDensitySRV = m_pShallowWater->GetDensitySRV();
			ID3D11ShaderResourceView* pBottomSRV = m_pShallowWater->GetBottomSRV();

			m_pImmediateContext->VSSetShaderResources(1,1, &pDensitySRV);
			m_pImmediateContext->VSSetShaderResources(2,1, &pBottomSRV);

			m_pImmediateContext->PSSetShaderResources(0,1, &pDensitySRV);	
			
			m_pImmediateContext->VSSetSamplers(1,1,&List::States::Get().m_pLinearSampler);
			List::States::Get().SetSamplers(m_pImmediateContext);

			m_pWaterSurface->Draw(m_pImmediateContext);

			DVector<ID3D11ShaderResourceView*> nullView(2);	
			m_pImmediateContext->PSSetShaderResources(0, 1, nullView.Begin());
			m_pImmediateContext->VSSetShaderResources(1, 2, nullView.Begin());

		}

		///< Draw Terrain
		{
			m_pPhong->SetDisplacement(m_pImmediateContext);

			m_pImmediateContext->VSSetSamplers(1,1,&List::States::Get().m_pLinearSampler);
			List::States::Get().SetSamplers(m_pImmediateContext);

			m_pTerrain->Draw(m_pImmediateContext);

		}

		{
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
void ParticleShallowWater::Release()
{	
	M::Delete(&m_pBench);

	M::Delete(&m_pShallowWater);

	M::Delete(&m_pQuad);	
	M::Delete(&m_pRawUV);
	M::Delete(&m_pPhong);
	M::Delete(&m_pWaterSurfaceShader);

	M::Delete(&m_pTerrain);
	M::Delete(&m_pWaterSurface);

	M::Delete(&m_pFxaa);
}


///<
void ParticleShallowWater::CreateMenu()
{
	TwBar* pBar = TwGetBarByIndex(1);

	if (pBar)
	{		
		TwAddVarRO(pBar, "FPS", TW_TYPE_FLOAT, &m_fps, " label='FPS' ");				
	}

	
	GPUSPHConstants::Get().CreateMenu();

	m_pShallowWater->CreateMenu();

	m_pPhong->CreateMenu();
}
