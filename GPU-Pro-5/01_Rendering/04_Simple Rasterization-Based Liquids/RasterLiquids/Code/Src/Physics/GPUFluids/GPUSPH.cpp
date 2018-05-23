
#include <Physics/GPUFluids/GPUSPH.hpp>
#include <Math/Matrix/Matrix.hpp>

#include <Graphics/Camera/Camera.hpp>
#include <Graphics/MeshImport.hpp>
#include <Graphics/Dx11/Mesh.hpp>

///<
void GPUSPH::SplatParticles(ID3D11DeviceContext* _pImmediateContext)
{
	_pImmediateContext->OMSetDepthStencilState(List::States::Get().m_pNoDepth, 0);

	_pImmediateContext->RSSetState(List::States::Get().m_pNoCullBackRasterizer);	

	float BlackColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f }; 
	_pImmediateContext->OMSetRenderTargets(1, &m_Up._pRTV, NULL);
	_pImmediateContext->ClearRenderTargetView(m_Up._pRTV, BlackColor);

	m_pSplat->Set(_pImmediateContext);

	_pImmediateContext->OMSetDepthStencilState(List::States::Get().m_pNoDepth, 0);
	_pImmediateContext->PSSetSamplers(0,1,&List::States::Get().m_pPointSampler);
	_pImmediateContext->PSSetSamplers(1,1,&List::States::Get().m_pLinearSampler);
	_pImmediateContext->RSSetState(List::States::Get().m_pNoCullBackRasterizer);	

	_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	float32 AddFactor[4]={1.0f, 1.0f, 1.0f, 1.0f};
	_pImmediateContext->OMSetBlendState(List::States::Get().m_pAddBlendState, AddFactor, 0xffffffff);
	
	uint32 offset = 0;
	uint32 stride=sizeof(Particle);
	_pImmediateContext->IASetVertexBuffers(0, 1, &m_pParticlesVertexBuffer[0], &stride, &offset );

	D3D11_VIEWPORT vp; 
	vp.MinDepth = 0.0f;   vp.MaxDepth = 1.0f;  vp.TopLeftX = 0;   vp.TopLeftY = 0;
	vp.Width	= (float32)GetDims().x();  	vp.Height	= (float32)GetDims().y();
	_pImmediateContext->RSSetViewports(1, &vp);

	_pImmediateContext->OMSetRenderTargets(1, &m_Up._pRTV, NULL);
	_pImmediateContext->Draw(m_iCurrentParticles,0);
	Dx11Renderer::UnbindResources(_pImmediateContext,0,1);
}

///<
void GPUSPH::AdvanceParticles(ID3D11DeviceContext* _pImmediateContext, Texture3D_SV& _correctedField)
{

	_pImmediateContext->IASetInputLayout(m_pSPHParticleLayout);

	_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
	_pImmediateContext->VSSetShader(m_pAdvanceParticlesVS, NULL, 0);
	_pImmediateContext->GSSetShader(m_pAdvanceParticlesGS, NULL, 0);

	_pImmediateContext->VSSetSamplers(0,1,&List::States::Get().m_pPointSampler);
	_pImmediateContext->VSSetSamplers(1,1,&List::States::Get().m_pLinearSampler);

	_pImmediateContext->PSSetShader(NULL, NULL, 0);
	uint32 stride = sizeof(Particle);
	uint32 offset = 0;
	_pImmediateContext->IASetVertexBuffers(0, 1, &m_pParticlesVertexBuffer[0], &stride, &offset);
	_pImmediateContext->SOSetTargets(1, &m_pParticlesVertexBuffer[1], &offset);	

	_pImmediateContext->VSSetShaderResources(0,1,&m_Up._pSRV);
	_pImmediateContext->VSSetShaderResources(1,1,&_correctedField._pSRV);
	_pImmediateContext->Draw(m_iCurrentParticles,0);

	ID3D11Buffer* pBuffers[1]={NULL};
	_pImmediateContext->SOSetTargets(1, pBuffers, &offset);
	std::swap(m_pParticlesVertexBuffer[0],m_pParticlesVertexBuffer[1]);
	Dx11Renderer::UnbindResources(_pImmediateContext, 0, 2);

}



///<
void GPUSPH::SetGridViewPort(ID3D11DeviceContext* _pImmediateContext)
{
	D3D11_VIEWPORT vp; 
	vp.MinDepth = 0.0f;   vp.MaxDepth = 1.0f;  vp.TopLeftX = 0;   vp.TopLeftY = 0;
	vp.Width	= (float32)GetDims().x();  	vp.Height	= (float32)GetDims().y();
	_pImmediateContext->RSSetViewports(1, &vp);
}

///<
void GPUSPH::Update(ID3D11DeviceContext* _pImmediateContext, ID3D11RenderTargetView* _pRT,  ID3D11DepthStencilView* _pDSV, Vector2ui _screenDims)
{

	static bool OneTwo=false;
	if (OneTwo)
	{
		OneTwo=false;
		if (m_iCurrentParticles<m_iNumParticles)
			m_iCurrentParticles= M::Min(m_iCurrentParticles+M::Cubed(m_iSqrtPerStep), m_iNumParticles);
	}
	else
		OneTwo=true;


	if (m_iCurrentParticles<m_iNumParticles)
		m_iCurrentParticles= M::Min(m_iCurrentParticles+M::Cubed(m_iSqrtPerStep), m_iNumParticles);
	
	ID3D11Buffer* _pBuffer = GPUSPHConstants::Get().GetConstantsBuffer();
	_pImmediateContext->GSSetConstantBuffers(3, 1, &_pBuffer);
	_pImmediateContext->PSSetConstantBuffers(3, 1, &_pBuffer);
	_pImmediateContext->VSSetConstantBuffers(3, 1, &_pBuffer);

	SplatParticles(_pImmediateContext);	
	
	{
		uint32 stride = sizeof(VolumeVertex);
		uint32 offset = 0;
		_pImmediateContext->IASetVertexBuffers(0, 1, &m_pVolumeVertexBuffer, &stride, &offset);

		_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		_pImmediateContext->IASetInputLayout(m_pVolumeLayout);

		_pImmediateContext->OMSetBlendState(List::States::Get().m_pDefaultNoBlendState, 0, 0xffffffff);
		_pImmediateContext->VSSetShader(m_pVolumeSlicesVS, NULL, 0);
		_pImmediateContext->GSSetShader(m_pVolumeSlicesGS, NULL, 0);
		
		SetGridViewPort(_pImmediateContext);

		///< Add SPH Gradient, forces (gravity).	
		{
			_pImmediateContext->PSSetShaderResources(0,1,&m_Up._pSRV);	
			_pImmediateContext->PSSetShaderResources(3,1,&m_P[0]._pSRV);

			if (m_pUserWaterHeightFieldSRV)
				_pImmediateContext->PSSetShaderResources(4,1,&m_pUserWaterHeightFieldSRV);	
			else
				_pImmediateContext->PSSetShaderResources(4,1,&m_pWaterHeightFieldSRV);	

			_pImmediateContext->OMSetRenderTargets(1, &m_UpCorrected._pRTV, NULL);
			_pImmediateContext->PSSetShader(m_pAddDensityGradientPS, NULL, 0);

			_pImmediateContext->Draw(6*GetDims().z(),0);
			Dx11Renderer::UnbindResources(_pImmediateContext, 0, 4);
		}

		///< FLIP
		if (m_bCreateDivergenceFree && m_bUseJacobi)
		{
			///< Compute Divergence		
			{
				ComputeDivergence(_pImmediateContext, m_UpCorrected);
			}

			///< Perform Jacobi Iterations
			{
				JacobiIterations(_pImmediateContext,20);
			}

			///< Make Divergence Free
			{
				_pImmediateContext->PSSetShader(m_pAddPressureGradientPS, NULL, 0);

				_pImmediateContext->PSSetShaderResources(3,1,&m_P[1]._pSRV);	
				_pImmediateContext->PSSetShaderResources(0,1,&m_UpCorrected._pSRV);	

				_pImmediateContext->OMSetRenderTargets(1, &m_UpCorrectedFLIP._pRTV, NULL);
				_pImmediateContext->Draw(6*GetDims().z(),0);
				//_pImmediateContext->OMSetBlendState(List::States::Get().m_pDefaultNoBlendState, AddFactor, 0xffffffff);
				Dx11Renderer::UnbindResources(_pImmediateContext, 0, 4);
			}
		}
	}
	
	///< Advance
	{
		if (m_bCreateDivergenceFree && m_bUseJacobi)
			AdvanceParticles(_pImmediateContext, m_UpCorrectedFLIP);		
		else
			AdvanceParticles(_pImmediateContext, m_UpCorrected);		
	}
		
	D3D11_VIEWPORT vp;
	vp.MinDepth = 0.0f;   vp.MaxDepth = 1.0f;  vp.TopLeftX = 0;   vp.TopLeftY = 0;
	vp.Width	= (float32)_screenDims.x();    
	vp.Height	= (float32)_screenDims.y();
	_pImmediateContext->RSSetViewports(1, &vp);
	_pImmediateContext->OMSetRenderTargets(1,&_pRT,_pDSV);
	_pImmediateContext->GSSetShader(NULL, NULL, 0);

}

///<
void GPUSPH::DrawParticles(ID3D11DeviceContext* _pImmediateContext)
{
	ASSERT(m_pRenderParticles!=NULL, "Not initialized)");

	ID3D11Buffer* _pBuffer=GPUSPHConstants::Get().GetConstantsBuffer();
	_pImmediateContext->GSSetConstantBuffers(3, 1, &_pBuffer);
	_pImmediateContext->PSSetConstantBuffers(3, 1, &_pBuffer);
	_pImmediateContext->VSSetConstantBuffers(3, 1, &_pBuffer);

	///< Set Normals

	m_pRenderParticles->Set(_pImmediateContext);
	_pImmediateContext->IASetInputLayout(m_pSPHParticleLayout);

	_pImmediateContext->RSSetState(List::States::Get().m_pCullBackRasterizer);
	float32 blendFact[4]={1,1,1,1};
	_pImmediateContext->OMSetBlendState(List::States::Get().m_pDefaultNoBlendState, blendFact, 0xffffffff);
	_pImmediateContext->OMSetDepthStencilState(List::States::Get().m_pDepthLess, 0);
	_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	uint32 stride = sizeof(Particle);
	uint32 offset = 0;
	_pImmediateContext->IASetVertexBuffers(0, 1, &m_pParticlesVertexBuffer[0], &stride, &offset);
	_pImmediateContext->Draw(m_iCurrentParticles,0); //
	_pImmediateContext->GSSetShader(NULL, NULL, 0);


}


///<
void GPUSPH::CreateParticles(ID3D11Device* _pDevice, const Vector3ui _dims, const uint32 _iParticlesPerAxis)
{

	m_iSqrtPerStep=0;

	m_iCurrentParticles=0;

	m_World = Matrix4f::Identity();

	
	uint32 iRatio = _dims.x()/_dims.z();
	uint32 iPerAxis=_iParticlesPerAxis;
	m_iParticlesPerAxis=Vector3ui(iPerAxis,iPerAxis,iPerAxis/iRatio);
	
	m_iNumParticles = m_iParticlesPerAxis.x()*m_iParticlesPerAxis.y()*m_iParticlesPerAxis.z();

	D3D11_BUFFER_DESC bd;
	memset(&bd,0,sizeof(D3D11_BUFFER_DESC));

	bd.Usage			= D3D11_USAGE_DEFAULT;
	bd.ByteWidth		= sizeof(Particle)*m_iNumParticles;

	bd.BindFlags		= D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_STREAM_OUTPUT | D3D11_BIND_SHADER_RESOURCE;

	Particle* pVertices = new Particle[m_iNumParticles];
	memset(pVertices,0,sizeof(Particle)*m_iNumParticles);

	Vector3f fDims(_dims);
	Vector3f dx = Vector3f(1.0f)/fDims;	

	if (m_bCreateJet)
	{
		Vector3f baseX					= Vector3f(0.8f,0.1f,0.2f);

		m_iSqrtPerStep=5;

		Vector3f particleSpacing		= DirectProduct(Vector3f(0.2f), Vector3f(1.0f/(float32)m_iSqrtPerStep));


		for(uint32 l=0; l<m_iNumParticles/M::Cubed(m_iSqrtPerStep);++l)
		{
			for (uint32 k=0; k<m_iSqrtPerStep; ++k)
			{
				for (uint32 i=0; i<m_iSqrtPerStep; ++i)
				{
					for (uint32 j=0; j<m_iSqrtPerStep; ++j)
					{
						int32 index = l*M::Cubed(m_iSqrtPerStep) + k*m_iSqrtPerStep*m_iSqrtPerStep + i*m_iSqrtPerStep + j;

						Vector3f posI = baseX + DirectProduct(Vector3f(M::SCast<float32>(i), M::SCast<float32>(j), M::SCast<float32>(k)), particleSpacing);

						pVertices[index].m_x = Vector4f(posI);

						pVertices[index].m_x[3]=1.0f;

						pVertices[index].m_data[Particle::U]=-0.8f;
						pVertices[index].m_data[Particle::V]=0;
						pVertices[index].m_data[Particle::W]=0.0f;
						pVertices[index].m_data[Particle::p]=0.8f;
					}
				}
			}
		}
	}
	else
	{
		m_iCurrentParticles=m_iNumParticles;
		m_iSqrtPerStep=0;

		Vector3f particleSpacing		= Vector3f(0.6f)/m_iParticlesPerAxis;
		Vector3f baseX					= Vector3f(0.2f,0.2f,0.3f);

		FOR_EACH_VECTOR3_COMPONENT(m_iParticlesPerAxis)
		{
			uint32 index = i + j*m_iParticlesPerAxis.x() + k*m_iParticlesPerAxis.x()*m_iParticlesPerAxis.y();
			Vector3f posI = baseX + DirectProduct(Vector3f(M::SCast<float32>(i), M::SCast<float32>(j), M::SCast<float32>(k)), particleSpacing);

			pVertices[index].m_x =Vector4f(posI);

			pVertices[index].m_x[3]=1.0f;

			pVertices[index].m_data[Particle::U]=0;
			pVertices[index].m_data[Particle::V]=0;
			pVertices[index].m_data[Particle::W]=0;
			pVertices[index].m_data[Particle::p]=0;
		}
		END_FOR_EACH_V
	}
	
	D3D11_SUBRESOURCE_DATA InitData;
	memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
	InitData.pSysMem = pVertices;

	HRESULT hr = _pDevice->CreateBuffer(&bd, &InitData, &m_pParticlesVertexBuffer[0]);
	ASSERT(hr==S_OK, "Failed to create Vertex Buffer!  ");
	hr = _pDevice->CreateBuffer(&bd, &InitData, &m_pParticlesVertexBuffer[1]);
	ASSERT(hr==S_OK, "Failed to create Vertex Buffer!  ");

	M::DeleteArray(&pVertices);
}

///<
void GPUSPH::Create(ID3D11Device* _pDevice, ID3D11DeviceContext* _pImmediateContext, const Vector3ui _dims, const uint32 _iParticlesPerAxis, const char* _strGround)
{
	m_iDims = _dims;

	m_bUseJacobi=false;
	m_bCreateDivergenceFree=false;
	m_bCreateJet=true;

	{
		const char* csShaderName = "..\\..\\Src\\Graphics\\Shaders\\Physics\\SPH.fx";
		// Define the input layout
		D3D11_INPUT_ELEMENT_DESC layout[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, sizeof(Vector4f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
		};	

		Dx11Renderer::CreateInputLayout(csShaderName,"VS_VolumeSim", _pDevice,layout, ARRAYSIZE(layout), &m_pVolumeLayout);
		Dx11Renderer::CreateVertexShader(csShaderName,"VS_VolumeSim", _pDevice, &m_pVolumeSlicesVS);
		Dx11Renderer::CreateGeometryShader(csShaderName,"GS_ARRAY", _pDevice, &m_pVolumeSlicesGS);

		Dx11Renderer::CreatePixelShader(csShaderName,"PS_AddDensityGradient", _pDevice, &m_pAddDensityGradientPS);

		///< FLIP
		if (m_bCreateDivergenceFree)
		{
			Dx11Renderer::CreatePixelShader(csShaderName,"PS_ComputeDivergence", _pDevice, &m_pComputeDivergencePS);
			Dx11Renderer::CreatePixelShader(csShaderName,"PS_Jacobi", _pDevice, &m_pJacobiPS);
			Dx11Renderer::CreatePixelShader(csShaderName,"PS_AddPressureGradient", _pDevice, &m_pAddPressureGradientPS);
		}

		Dx11Renderer::CreateVertexShader(csShaderName,"VS_AdvanceParticles", _pDevice, &m_pAdvanceParticlesVS);

		m_pRenderParticles=new Shader();
		Dx11Renderer::CreateShadersAndLayout(csShaderName, "VS_RenderParticles", "PS_RenderParticles", "GS_RenderParticles", layout, ARRAYSIZE(layout), m_pRenderParticles, _pDevice);		
	}

	{		
		const char* csShaderName = "..\\..\\Src\\Graphics\\Shaders\\Physics\\SPH.fx";

		///< Define the input layout.
		D3D11_INPUT_ELEMENT_DESC layout[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, sizeof(Vector4f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
		};		

		Dx11Renderer::CreateInputLayout(csShaderName,"VS_AdvanceParticles", _pDevice,layout, ARRAYSIZE(layout), &m_pSPHParticleLayout);

		ID3DBlob* pVSBlob = NULL;
		HRESULT hr = Dx11Renderer::CompileShaderFromFile(csShaderName, "VS_AdvanceParticles", Dx11Renderer::VSLevel(_pDevice), &pVSBlob);
		ASSERT(hr==S_OK, "Failed to create shader");	

		D3D11_SO_DECLARATION_ENTRY pDecl[] =
		{
			// stream, semantic name, semantic index, start component, component count, output slot
			{ 0, "POSITION", 0, 0, 4, 0 },   
			{ 0, "TEXCOORD", 0, 0, 4, 0 }
		};

		uint32 strides[1]={sizeof(pDecl)};	
		hr = _pDevice->CreateGeometryShaderWithStreamOutput(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), pDecl, 2, strides, 1, 0, 
			NULL, &m_pAdvanceParticlesGS) ;
		
		ASSERT(hr==S_OK, "Failed Creating Geometry Stream");
		M::Release(&pVSBlob);

		{
			const char* csShaderName = "..\\..\\Src\\Graphics\\Shaders\\Physics\\SPHSplatting.fx";		
			m_pSplat = new Shader();
			///< Define the input layout.
			D3D11_INPUT_ELEMENT_DESC layout[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, sizeof(Vector4f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
			};		

			Dx11Renderer::CreateShadersAndLayout(csShaderName, "Splatting_VS", "Splatting_PS", "Splatting_GS", layout, ARRAYSIZE(layout), m_pSplat, _pDevice);	
		}

	}

	m_pVolumeVertexBuffer = CubeUV::CreateCubeUV(_pDevice, _dims);

	CreateVolumeTextures(_pDevice, _dims, DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_FLOAT);
	CreateParticles(_pDevice, _dims, _iParticlesPerAxis);

	std::string baseTexture = "..\\..\\Ressources\\Textures\\";
	std::string tFileName = std::string(_strGround);
	if (tFileName.length()!=0)
	{
		HRESULT hr = D3DX11CreateShaderResourceViewFromFile(_pDevice, (baseTexture+tFileName).c_str(), NULL, NULL, &m_pWaterHeightFieldSRV, NULL );
		ASSERT(hr ==S_OK, "Failed loading texture !");
	}
	
}

///<
GPUSPH::~GPUSPH()
{
	M::Release(&m_pWaterHeightFieldSRV);

	for(int32 i=0; i<2;++i)
	{
		M::Release(&m_P[i]);	
		M::Release(&m_pParticlesVertexBuffer[i]);
	}



	M::Release(&m_Up);
	M::Release(&m_UpCorrected);
	M::Release(&m_UpCorrectedFLIP);

	M::Release(&m_Div);	

	M::Release(&m_pVolumeSlicesVS);	
	M::Release(&m_pVolumeSlicesGS);	
			
	M::Delete(&m_pRenderParticles);

	M::Delete(&m_pSplat);
	M::Release(&m_pAddDensityGradientPS);
	M::Release(&m_pJacobiPS);
	M::Release(&m_pAddPressureGradientPS);
	
	M::Release(&m_pComputeDivergencePS);

	M::Release(&m_pSPHParticleLayout);
	M::Release(&m_pAdvanceParticlesVS);
	M::Release(&m_pAdvanceParticlesGS);
		
	M::Release(&m_pVolumeLayout);	
	///< Volume Constants ?

	M::Release(&m_pVolumeVertexBuffer);
}


GPUSPHConstants GPUSPHConstants::m_instance;
///<
void GPUSPHConstants::CreateMenu()
{
	TwBar* pBar = MenuManager::Get().AddBar("SPH", 7);
	if (pBar)
	{		
		TwAddVarRW(pBar, "PIC-FLIP Ratio", TW_TYPE_FLOAT, &m_constants._PIC_FLIP, " min=0.05 max=1.0 step=0.05");
		
		TwAddVarRW(pBar, "Pressure Scale", TW_TYPE_FLOAT, &m_constants._PressureScale, " min=5.0 max=7.0 step=0.5");
		TwAddVarRW(pBar, "Initial Density", TW_TYPE_FLOAT, &m_constants._InitialDensity, " min=7.0 max=10.0 step=0.1");	
		
		TwAddVarRW(pBar, "Surface Density", TW_TYPE_FLOAT, &m_constants._SurfaceDensity, " min=0.2 max=1.5 step=0.05");
		TwAddVarRW(pBar, "Time Step", TW_TYPE_FLOAT, &m_constants._TimeStep, " min=0.001 max=0.04 step=0.001");

		TwAddVarRW(pBar, "Gravity Direction", TW_TYPE_DIR3F, &m_constants._Gravity, "");
	}
}

///<
void GPUSPHConstants::UpdateConstants(ID3D11DeviceContext* _pImmediateContext)
{	
	_pImmediateContext->UpdateSubresource(m_pConstants, 0, NULL, &m_constants, 0, 0);
}

///<
void GPUSPHConstants::CreateContants(ID3D11Device* _pDevice, Vector3i _GridDims)
{
	///<	
	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));

	///< Create the constant buffers
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(GPUSPHConstants::SPHContants);
	bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	bd.CPUAccessFlags = 0;
	HRESULT hr = _pDevice->CreateBuffer(&bd, NULL, &m_pConstants);
	ASSERT(hr==S_OK, "Failed To Creat Constant Buffer");	

	for(int32 i=0; i<3; ++i)
		m_constants._GridSpacing[i] = 1.0f/M::SCast<float32>(_GridDims[i]);

	///< Default Values
	m_constants._Gravity=Vector4f(0,1.0f,0,0);
	m_constants._InitialDensity[0]=8.5f;
	m_constants._PressureScale=6.0f;
	m_constants._PIC_FLIP=0.05f;
	m_constants._SurfaceDensity=0.35f;
	m_constants._TimeStep=0.035f;

}

void GPUSPH::CreateMenu()
{
	TwBar* pBar = TwGetBarByIndex(1);

	if (pBar)
	{	
		TwAddVarRO(pBar, "Num Particles", TW_TYPE_INT32, &m_iNumParticles, " label='Num Particles' ");
	}

}

///<
void GPUSPH::CreateVolumeTextures(ID3D11Device* _pDevice, const Vector3ui _dims, DXGI_FORMAT _VectorFormat, DXGI_FORMAT _ScalarFormat)
{
	Vector1f* pScalarPixels = new Vector1f[_dims.x()*_dims.y()*_dims.z()];
	memset(pScalarPixels, 0, sizeof(Vector1f)*_dims.x()*_dims.y()*_dims.z());

	Vector4f* pGradientPixels = new Vector4f[_dims.x()*_dims.y()*_dims.z()];	
	memset(pGradientPixels, 0, sizeof(Vector4f)*_dims.x()*_dims.y()*_dims.z());

	///< Gradient
	D3D11_SUBRESOURCE_DATA GradientInitialData;
	GradientInitialData.pSysMem			= reinterpret_cast<float32*>(pGradientPixels);
	GradientInitialData.SysMemPitch		= _dims.x()*sizeof(Vector4f);
	GradientInitialData.SysMemSlicePitch	= _dims.y()*GradientInitialData.SysMemPitch;

	m_Up=Dx11Renderer::Create3DTexture(_pDevice,_VectorFormat, _dims, &GradientInitialData);
	m_UpCorrected=Dx11Renderer::Create3DTexture(_pDevice,_VectorFormat, _dims, &GradientInitialData);	
	
	if (m_bCreateDivergenceFree)
	{
		D3D11_SUBRESOURCE_DATA ScalaraInitialData;
		ScalaraInitialData.pSysMem			= reinterpret_cast<float32*>(pScalarPixels);
		ScalaraInitialData.SysMemPitch		= _dims.x()*sizeof(Vector1f);
		ScalaraInitialData.SysMemSlicePitch	= _dims.y()*ScalaraInitialData.SysMemPitch;

		m_UpCorrectedFLIP=Dx11Renderer::Create3DTexture(_pDevice,_VectorFormat, _dims, &GradientInitialData);	
		m_Div=Dx11Renderer::Create3DTexture(_pDevice,_ScalarFormat, _dims, &ScalaraInitialData);

		for (int32 i=0; i<2; ++i)
			m_P[i]=Dx11Renderer::Create3DTexture(_pDevice,_ScalarFormat, _dims, &ScalaraInitialData);

	}

	M::DeleteArray(&pGradientPixels);
	M::DeleteArray(&pScalarPixels);
}

///<
void GPUSPH::JacobiIterations(ID3D11DeviceContext* _pImmediateContext, const uint32 _uiIterations)
{
	_pImmediateContext->PSSetShader(m_pJacobiPS, NULL, 0);
	_pImmediateContext->PSSetShaderResources(0, 1, &m_Up._pSRV);
	_pImmediateContext->PSSetShaderResources(2, 1, &m_Div._pSRV);

	Vector4f zero(0);
	_pImmediateContext->ClearRenderTargetView(m_P[0]._pRTV,zero.Begin());
	_pImmediateContext->ClearRenderTargetView(m_P[1]._pRTV,zero.Begin());

	for (uint32 i=0; i<_uiIterations; ++i)
	{
		uint32 uiCurrent=i%2;
		uint32 uiNext=(i+1)%2;
		_pImmediateContext->PSSetShaderResources(3, 1, &m_P[uiCurrent]._pSRV);
		_pImmediateContext->OMSetRenderTargets(1, &m_P[uiNext]._pRTV, NULL);

		_pImmediateContext->Draw(6*GetDims().z(),0);

		ID3D11ShaderResourceView* nullViews[1] = {0};
		_pImmediateContext->PSSetShaderResources(3, 1, nullViews);
		_pImmediateContext->OMSetRenderTargets(0, 0, 0); 
	}

	ID3D11ShaderResourceView* nullViews[1] = {0};
	_pImmediateContext->PSSetShaderResources(2, 1, nullViews);
	_pImmediateContext->PSSetShaderResources(0, 1, nullViews);

}

///<
void GPUSPH::ComputeDivergence(ID3D11DeviceContext* _pImmediateContext, Texture3D_SV& _vel)
{
	_pImmediateContext->PSSetShader(m_pComputeDivergencePS, NULL, 0);

	_pImmediateContext->PSSetShaderResources(0,1,&_vel._pSRV);	
	_pImmediateContext->OMSetRenderTargets(1, &m_Div._pRTV, NULL);
	_pImmediateContext->Draw(6*GetDims().z(),0);
	Dx11Renderer::UnbindResources(_pImmediateContext, 0, 1);
}
