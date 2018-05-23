

#include <Apps/GPULiquids/GPU2DParticleLiquids.hpp>

#include <Graphics/Camera/Camera.hpp>
#include <Graphics/Camera/Dx11Camera.hpp>
#include <Graphics/Dx11/Post/FXAA.hpp>
#include <Graphics/Dx11/Utility/Dx11Benchmarks.hpp>///<
#include <Graphics/Dx11/Shaders.hpp>
#include <Graphics/Dx11/Mesh.hpp>

#include <Physics/GPUFluids/GPUSPH2D.hpp>

///<
void Simple2DLiquids::CreatePostQuad()
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

			CreateShadersAndLayout(csRawUVName, "PostVS", "RenderDensityField_PS", NULL, layout, ARRAYSIZE(layout), m_pRawUV, m_pDevice);
		}		

	}


	m_pQuad = new QuadUV();
	m_pQuad->Create(m_pDevice, NULL);

	std::string baseTexture = "..\\..\\Ressources\\Textures\\";

	std::string strNormalsTextureFileName = std::string("ParticleMask.png");
	if (strNormalsTextureFileName.length()!=0)
	{
		HRESULT hr = D3DX11CreateShaderResourceViewFromFile(m_pDevice, (baseTexture+strNormalsTextureFileName).c_str(), NULL, NULL, &m_pParticleMask, NULL );
		ASSERT(hr ==S_OK, "Failed loading texture !");
	}

}

///<
bool Simple2DLiquids::Create(HWND _hWnd)
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
			m_p2DSPH = new GPUSPH2D();
			m_p2DSPH->Create(m_pDevice,m_pImmediateContext, Vector2i(256), 256);//+128
		}

		CreatePostQuad();

		return true;
	}
	return false;
}


///<
void Simple2DLiquids::CreateContants(ID3D11Device* _pDevice)
{

}

///<
bool Simple2DLiquids::Update()
{
	///< 
	float32 dt=1.0f/30.0f;

	m_pBench->Begin(m_pImmediateContext);
	
	ID3D11RenderTargetView* pCurrentRT = m_pRenderTargetView;
	if (m_pFxaa)
		pCurrentRT = m_pFxaa->GetRenderTarget();	  

	///< Render Scene
	{
		float ClearColor[4] = { 0.25f, 0.25f, 0.25f, 0.0f }; 
		m_pImmediateContext->ClearRenderTargetView(pCurrentRT, ClearColor);
		m_pImmediateContext->ClearDepthStencilView(m_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0 );

		Camera::Get().Update();
		m_pCamera->SetParams(m_pImmediateContext,0);

		m_p2DSPH->Update(m_pImmediateContext,pCurrentRT,m_pDepthStencilView, Vector2i(m_w,m_h));
		
		m_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		m_pImmediateContext->OMSetBlendState(List::States::Get().m_pDefaultNoBlendState,  0, 0xffffffff);
		m_pImmediateContext->RSSetState(List::States::Get().m_pNoCullBackRasterizer);	
		m_pImmediateContext->OMSetDepthStencilState(List::States::Get().m_pDepthLess, 0);

		m_pImmediateContext->OMSetRenderTargets(1, &pCurrentRT, m_pDepthStencilView);

		D3D11_VIEWPORT vp;
		vp.MinDepth = 0.0f;   vp.MaxDepth = 1.0f;  vp.TopLeftX = 0;   vp.TopLeftY = 0;
		vp.Width	= (float32)m_w;    		vp.Height	= (float32)m_h;
		m_pImmediateContext->RSSetViewports(1, &vp);

		m_p2DSPH->DrawParticles(m_pImmediateContext, m_pParticleMask);

		{
			///< GUI
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
void Simple2DLiquids::Release()
{	

	m_up.Release();

	M::Delete(&m_pBench);

	M::Delete(&m_p2DSPH);

	M::Delete(&m_pQuad);	
	M::Delete(&m_pRawUV);

	M::Delete(&m_pFxaa);
}


///<
void Simple2DLiquids::CreateMenu()
{
	TwBar* pBar = TwGetBarByIndex(1);

	if (pBar)
	{		

		TwAddVarRO(pBar, "FPS", TW_TYPE_FLOAT, &m_fps, " label='FPS' ");		
		

		m_p2DSPH->CreateMenu();
		
	}
}
