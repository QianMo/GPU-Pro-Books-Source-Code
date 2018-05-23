
#include <Physics/GPUFluids/GPUSPH2D.hpp>

#include <Math/Matrix/Matrix.hpp>


#include <Graphics/Camera/Camera.hpp>

#include <Graphics/MeshImport.hpp>
#include <Graphics/Dx11/Mesh.hpp>

///<
void GPUSPH2D::SplatParticles(ID3D11DeviceContext* _pImmediateContext)
{
	m_pSplat->Set(_pImmediateContext);

	_pImmediateContext->GSSetConstantBuffers(3, 1, &m_pConstants);
	_pImmediateContext->PSSetConstantBuffers(3, 1, &m_pConstants);
	_pImmediateContext->VSSetConstantBuffers(3, 1, &m_pConstants);

	_pImmediateContext->OMSetDepthStencilState(List::States::Get().m_pNoDepth, 0);
	_pImmediateContext->PSSetSamplers(0,1,&List::States::Get().m_pPointSampler);
	_pImmediateContext->PSSetSamplers(1,1,&List::States::Get().m_pLinearSampler);
	_pImmediateContext->RSSetState(List::States::Get().m_pNoCullBackRasterizer);	

	Vector4f zero(0);
	_pImmediateContext->OMSetRenderTargets(1, &m_Up._pRTV, NULL);
	_pImmediateContext->ClearRenderTargetView(m_Up._pRTV, zero.Begin());
	_pImmediateContext->ClearRenderTargetView(m_UpCorrected._pRTV, zero.Begin());

	_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	float32 AddFactor[4]={1.0f, 1.0f, 1.0f, 1.0f};
	_pImmediateContext->OMSetBlendState(List::States::Get().m_pAddBlendState, AddFactor, 0xffffffff);
	
	uint32 offset = 0;
	uint32 stride = sizeof(Particle);
	_pImmediateContext->IASetVertexBuffers(0, 1, &m_pParticlesVertexBuffer[0], &stride, &offset );

	D3D11_VIEWPORT vp; 
	vp.MinDepth = 0.0f;   vp.MaxDepth = 1.0f;  vp.TopLeftX = 0;   vp.TopLeftY = 0;
	vp.Width	= (float32)GetDims().x();  	vp.Height	= (float32)GetDims().y();
	_pImmediateContext->RSSetViewports(1, &vp);

	_pImmediateContext->Draw(m_iNumParticles,0);
	Dx11Renderer::UnbindResources(_pImmediateContext,0,1);
}

///<
void GPUSPH2D::AddDensityGradient(ID3D11DeviceContext* _pImmediateContext)
{
	///< Add Gradients to U_corrected
	m_pAddDensityGradient->Set(_pImmediateContext);

	///< Set quad geometry, draw the quad and add the gradients.		
	_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	_pImmediateContext->OMSetBlendState(List::States::Get().m_pDefaultNoBlendState, 0, 0xffffffff);

	D3D11_VIEWPORT vp; 
	vp.MinDepth = 0.0f;   vp.MaxDepth = 1.0f;  vp.TopLeftX = 0;   vp.TopLeftY = 0;
	vp.Width	= (float32)GetDims().x();  	vp.Height	= (float32)GetDims().y();
	_pImmediateContext->RSSetViewports(1, &vp);

	///< Add SPH Gradient, forces (gravity).	
	{			
		_pImmediateContext->PSSetShaderResources(0,1,&m_Up._pSRV);	
		_pImmediateContext->OMSetRenderTargets(1, &m_UpCorrected._pRTV, NULL);			
		m_pPostQuad->Draw(_pImmediateContext);
		
		Dx11Renderer::UnbindResources(_pImmediateContext, 0, 1);

		if (m_bCreateDivergenceFree && m_bMenuUseJacobi)
		{
			_pImmediateContext->PSSetShader(m_pComputeDiv, NULL, 0);
			_pImmediateContext->PSSetShaderResources(0,1,&m_UpCorrected._pSRV);
			_pImmediateContext->OMSetRenderTargets(1, &m_Div._pRTV, NULL);
			m_pPostQuad->Draw(_pImmediateContext);
			Dx11Renderer::UnbindResources(_pImmediateContext, 0, 1);

			
			_pImmediateContext->PSSetShader(m_pJacobi, NULL, 0);

			_pImmediateContext->PSSetShaderResources(0,1,&m_Up._pSRV);
			_pImmediateContext->PSSetShaderResources(4,1,&m_Div._pSRV);
			
			Vector4f zero(0);
			_pImmediateContext->ClearRenderTargetView(m_P[0]._pRTV,zero.Begin());
			_pImmediateContext->ClearRenderTargetView(m_P[1]._pRTV,zero.Begin());

			for (int32 i=0; i<50; ++i)
			{
				uint32 uiCurrent=i%2;
				uint32 uiNext=(i+1)%2;

				_pImmediateContext->PSSetShaderResources(3,1,&m_P[uiCurrent]._pSRV);
				_pImmediateContext->OMSetRenderTargets(1, &m_P[uiNext]._pRTV, NULL);
				m_pPostQuad->Draw(_pImmediateContext);

				_pImmediateContext->OMSetRenderTargets(0, 0, 0); 
				DVector<ID3D11ShaderResourceView*> nullView(1);	
				_pImmediateContext->PSSetShaderResources(3, 1, nullView.Begin());
			}

			Dx11Renderer::UnbindResources(_pImmediateContext, 0, 1);
			DVector<ID3D11ShaderResourceView*> nullView(2);	
			_pImmediateContext->PSSetShaderResources(3, 2, nullView.Begin());

			///< Correct Finally
			_pImmediateContext->PSSetShader(m_pAddPressureGradient, NULL, 0);
			_pImmediateContext->PSSetShaderResources(3,1,&m_P[1]._pSRV);
			_pImmediateContext->PSSetShaderResources(0,1,&m_UpCorrected._pSRV);

			_pImmediateContext->OMSetRenderTargets(1, &m_FLIPCorrectedUp._pRTV, NULL);
			m_pPostQuad->Draw(_pImmediateContext);
			Dx11Renderer::UnbindResources(_pImmediateContext, 0, 2);
			_pImmediateContext->PSSetShaderResources(3, 1, nullView.Begin());



		}
	}	
}

///<
void GPUSPH2D::AdvanceParticles(ID3D11DeviceContext* _pImmediateContext, Texture2D_SV& _correctedField)
{

	_pImmediateContext->IASetInputLayout(m_pSPHParticleLayout);

	_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
	_pImmediateContext->VSSetShader(m_pAdvanceParticlesVS, NULL, 0);
	_pImmediateContext->GSSetShader(m_pAdvanceParticlesGS, NULL, 0);

	_pImmediateContext->GSSetConstantBuffers(3, 1, &m_pConstants);	
	_pImmediateContext->VSSetConstantBuffers(3, 1, &m_pConstants);

	_pImmediateContext->VSSetSamplers(0,1,&List::States::Get().m_pPointSampler);
	_pImmediateContext->VSSetSamplers(1,1,&List::States::Get().m_pLinearSampler);

	_pImmediateContext->PSSetShader(NULL, NULL, 0);
	uint32 stride = sizeof(Particle);
	uint32 offset = 0;
	_pImmediateContext->IASetVertexBuffers(0, 1, &m_pParticlesVertexBuffer[0], &stride, &offset);
	_pImmediateContext->SOSetTargets(1, &m_pParticlesVertexBuffer[1], &offset);	

	_pImmediateContext->VSSetShaderResources(0,1,&m_Up._pSRV);
	_pImmediateContext->VSSetShaderResources(1,1,&_correctedField._pSRV);
	_pImmediateContext->Draw(m_iNumParticles,0);

	ID3D11Buffer* pBuffers[1]={NULL};
	_pImmediateContext->SOSetTargets(1, pBuffers, &offset);
	std::swap(m_pParticlesVertexBuffer[0],m_pParticlesVertexBuffer[1]);
	Dx11Renderer::UnbindResources(_pImmediateContext, 0, 2);

}



///<
void GPUSPH2D::Update(ID3D11DeviceContext* _pImmediateContext, ID3D11RenderTargetView* _pRT,  ID3D11DepthStencilView* _pDSV, Vector2i _screenDims)
{

	UpdateConstants(_pImmediateContext);

	///< Draw particle density.
	SplatParticles(_pImmediateContext);	
	
	///< Compute Gradients
	AddDensityGradient(_pImmediateContext);

	if (m_bCreateDivergenceFree)
	{
		///< Compute Div;

		///< Add Pressure Gradients;
	}
	
	///< Advect Particles
	if (m_bCreateDivergenceFree && m_bMenuUseJacobi)
		AdvanceParticles(_pImmediateContext, m_FLIPCorrectedUp);	
	else
		AdvanceParticles(_pImmediateContext, m_UpCorrected);	

	D3D11_VIEWPORT vp;
	vp.MinDepth = 0.0f;   vp.MaxDepth = 1.0f;  vp.TopLeftX = 0;   vp.TopLeftY = 0;
	vp.Width	= (float32)_screenDims.x();    
	vp.Height	= (float32)_screenDims.y();
	_pImmediateContext->RSSetViewports(1, &vp);
	_pImmediateContext->OMSetRenderTargets(1,&_pRT,_pDSV);
	_pImmediateContext->GSSetShader(NULL, NULL, 0);

}

///<
void GPUSPH2D::DrawParticles(ID3D11DeviceContext* _pImmediateContext, ID3D11ShaderResourceView* _pMask)
{
	
	m_pRenderParticles->Set(_pImmediateContext);
	_pImmediateContext->IASetInputLayout(m_pSPHParticleLayout);

	_pImmediateContext->GSSetConstantBuffers(3, 1, &m_pConstants);	
	_pImmediateContext->VSSetConstantBuffers(3, 1, &m_pConstants);

	_pImmediateContext->RSSetState(List::States::Get().m_pNoCullBackRasterizer);				
	_pImmediateContext->OMSetDepthStencilState(List::States::Get().m_pNoDepth, 0);
	_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	_pImmediateContext->PSSetShaderResources(2,1,&_pMask);
	uint32 stride = sizeof(Particle);
	uint32 offset = 0;
	_pImmediateContext->IASetVertexBuffers(0, 1, &m_pParticlesVertexBuffer[0], &stride, &offset);
	_pImmediateContext->Draw(m_iNumParticles,0); //
	_pImmediateContext->GSSetShader(NULL, NULL, 0);

	DVector<ID3D11ShaderResourceView*> nullView(3);	
	_pImmediateContext->PSSetShaderResources(0, 3, nullView.Begin());

}

///<
void GPUSPH2D::CreateParticles(ID3D11Device* _pDevice, const Vector2i _dims, const int32 _iParticlesPerAxis)
{

	m_World = Matrix4f::Identity();

	m_iParticlesPerAxis = Vector2ui(_iParticlesPerAxis,_iParticlesPerAxis);
	
	m_iNumParticles = m_iParticlesPerAxis.x()*m_iParticlesPerAxis.y();

	D3D11_BUFFER_DESC bd;
	memset(&bd,0,sizeof(D3D11_BUFFER_DESC));

	bd.Usage			= D3D11_USAGE_DEFAULT;
	bd.ByteWidth		= sizeof(Particle)*m_iNumParticles;

	bd.BindFlags		= D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_STREAM_OUTPUT | D3D11_BIND_SHADER_RESOURCE;

	Particle* pVertices = new Particle[m_iNumParticles];
	memset(pVertices,0,sizeof(Particle)*m_iNumParticles);
	
	Vector2f fDims(_dims);

	Vector2f particleSpacing		= Vector2f(0.5f)/Vector2f(m_iParticlesPerAxis);
	Vector2f baseX					= Vector2f(0.25f, 0.25f);

	FOR_EACH_VECTOR2_COMPONENT(m_iParticlesPerAxis)
	{
		uint32 index = i + j*m_iParticlesPerAxis.x();
		Vector2f posI = baseX + DirectProduct(Vector2f(M::SCast<float32>(i), M::SCast<float32>(j)), particleSpacing);

		pVertices[index].m_x =Vector4f(posI.x(),posI.y(),0,1);

		pVertices[index].m_data[Particle::U]=0;
		pVertices[index].m_data[Particle::V]=0;
		pVertices[index].m_data[Particle::W]=0;
		pVertices[index].m_data[Particle::p]=1;
	}
	END_FOR_EACH

	D3D11_SUBRESOURCE_DATA InitData;
	memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
	InitData.pSysMem = pVertices;

	HRESULT hr = _pDevice->CreateBuffer(&bd, &InitData, &m_pParticlesVertexBuffer[0]);
	ASSERT(hr==S_OK, "Failed to create Vertex Buffer!  ");
	hr = _pDevice->CreateBuffer(&bd, NULL, &m_pParticlesVertexBuffer[1]);
	ASSERT(hr==S_OK, "Failed to create Vertex Buffer!  ");

	M::DeleteArray(&pVertices);
}

///<
void GPUSPH2D::Create(ID3D11Device* _pDevice, ID3D11DeviceContext* _pImmediateContext, const Vector2i _dims, const int32 _iParticlesPerAxis)
{
	m_bCreateDivergenceFree=true;
	m_bMenuUseJacobi=false;

	m_iDims = _dims;

	const char* csShaderName = "..\\..\\Src\\Graphics\\Shaders\\Physics\\SPH2D.fx";
	{
		
		// Define the input layout
		D3D11_INPUT_ELEMENT_DESC particleLayout[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, sizeof(Vector4f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
		};	

		m_pRenderParticles=new Shader();
		Dx11Renderer::CreateShadersAndLayout(csShaderName, "VS_RenderParticles", "PS_RenderParticles", "GS_RenderParticles", particleLayout, ARRAYSIZE(particleLayout), m_pRenderParticles, _pDevice);

		m_pSplat = new Shader();
		Dx11Renderer::CreateShadersAndLayout(csShaderName, "Splatting_VS", "Splatting_PS", "Splatting_GS", particleLayout, ARRAYSIZE(particleLayout), m_pSplat, _pDevice);	

		///< Particle Transport:
		Dx11Renderer::CreateInputLayout(csShaderName,"VS_AdvanceParticles", _pDevice, particleLayout, ARRAYSIZE(particleLayout), &m_pSPHParticleLayout);

		Dx11Renderer::CreateVertexShader(csShaderName,"VS_AdvanceParticles",_pDevice, &m_pAdvanceParticlesVS);

		///< Create Stream Geometry Shader
		{
			ID3DBlob* pVSBlob = NULL;
			HRESULT hr = Dx11Renderer::CompileShaderFromFile(csShaderName, "VS_AdvanceParticles", Dx11Renderer::VSLevel(_pDevice), &pVSBlob);
			ASSERT(hr==S_OK, "Failed to create shader");	


			D3D11_SO_DECLARATION_ENTRY pDecl[] =
			{
				// Stream, semantic name, semantic index, start component, component count, output slot
				{ 0, "POSITION", 0, 0, 4, 0 },   
				{ 0, "TEXCOORD", 0, 0, 4, 0 }
			};

			uint32 strides[1]={sizeof(pDecl)};	
			hr = _pDevice->CreateGeometryShaderWithStreamOutput(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), pDecl, 2, strides, 1, 0, 
				NULL, &m_pAdvanceParticlesGS) ;

			ASSERT(hr==S_OK, "Failed Creating Geometry Stream");
			M::Release(&pVSBlob);
		}
		
	}

	{
		D3D11_INPUT_ELEMENT_DESC postLayout[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, sizeof(Vector3f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
		};

		///< Quad Post Process
		m_pAddDensityGradient = new Shader();
		Dx11Renderer::CreateShadersAndLayout(csShaderName, "RenderQuad_VS", "PS_AddDensityGradient", NULL, postLayout, ARRAYSIZE(postLayout), m_pAddDensityGradient, _pDevice);

		if (m_bCreateDivergenceFree)
		{
			Dx11Renderer::CreatePixelShader(csShaderName, "PS_AddPressureGradient", _pDevice, &m_pAddPressureGradient);
			Dx11Renderer::CreatePixelShader(csShaderName, "PS_Jacobi", _pDevice, &m_pJacobi);
			Dx11Renderer::CreatePixelShader(csShaderName, "PS_ComputeDivergence", _pDevice, &m_pComputeDiv);
			
		}
	}
	
	CreateTextures(_pDevice, _dims);
	CreateParticles(_pDevice, _dims, _iParticlesPerAxis);
	CreateContants(_pDevice);
	
	m_pPostQuad=new QuadUV();
	m_pPostQuad->Create(_pDevice, NULL);
}

///<
void GPUSPH2D::CreateContants(ID3D11Device* _pDevice)
{
	///<	
	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));

	///< Create the constant buffers
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(SPHContants);
	bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	bd.CPUAccessFlags = 0;
	HRESULT hr = _pDevice->CreateBuffer(&bd, NULL, &m_pConstants);
	ASSERT(hr==S_OK, "Failed To Creat Constant Buffer");	

	for(int32 i=0; i<2; ++i)
		m_constants._GridSpacing[i] = 1.0f/M::SCast<float32>(m_iDims[i]);

	///< Default Values
	m_constants._Gravity=Vector4f(0,0.65f,0,0);
	m_constants._InitialDensity[0]=5.0f;
	m_constants._PressureScale=1.0f;
	m_constants._PIC_FLIP=0.05f;
	m_constants._SurfaceDensity=0.35f;
	m_constants._TimeStep=0.004f;

}

///<
void GPUSPH2D::CreateMenu()
{
	TwBar* pBar = MenuManager::Get().AddBar("2D SPH", 9);
	if (pBar)
	{		
		TwAddVarRO(pBar, "Num Particles", TW_TYPE_INT32, &m_iNumParticles, " label='Num Particles' ");
		TwAddVarRO(pBar, "Grid Dim", TW_TYPE_INT32, &m_iDims[0], " label='Grid Dims' ");

		TwAddVarRW(pBar, "PIC-FLIP Ratio", TW_TYPE_FLOAT, &m_constants._PIC_FLIP, " min=0.03 max=1.0 step=0.01");
		TwAddVarRW(pBar, "Initial Density", TW_TYPE_FLOAT, &m_constants._InitialDensity[0], " min=4.5 max=12.0 step=0.1");
		TwAddVarRW(pBar, "Pressure Scale", TW_TYPE_FLOAT, &m_constants._PressureScale, " min=0.5 max=6.0 step=0.1");
		TwAddVarRW(pBar, "Time Step", TW_TYPE_FLOAT, &m_constants._TimeStep, " min=0.001 max=0.007 step=0.001");

		TwAddVarRW(pBar, "Use Jacobi?", TW_TYPE_BOOL8, &m_bMenuUseJacobi, "");
		TwAddVarRW(pBar, "Gravity Direction", TW_TYPE_DIR3F, &m_constants._Gravity, "");
	}
}

///<
void GPUSPH2D::UpdateConstants(ID3D11DeviceContext* _pImmediateContext)
{	
	_pImmediateContext->UpdateSubresource(m_pConstants, 0, NULL, &m_constants, 0, 0);
}


///<
void GPUSPH2D::CreateTextures(ID3D11Device* _pDevice, const Vector2i _dims)
{
	
	Vector1f* pScalarPixels = new Vector1f[_dims.x()*_dims.y()];
	memset(pScalarPixels, 0, sizeof(Vector1f)*_dims.x()*_dims.y());
	Vector4f* pGradientPixels = new Vector4f[_dims.x()*_dims.y()];
	memset(pGradientPixels, 0, sizeof(Vector4f)*_dims.x()*_dims.y());
/*
	FOR_EACH_VECTOR2_COMPONENT(_dims)
	{
		pGradientPixels[_dims.x()*j + i]=Vector4f(0,0,0,0);
	}
	END_FOR_EACH
	*/

	///< Check first 3D.
	D3D11_SUBRESOURCE_DATA GradientInitialData;
	GradientInitialData.pSysMem				= reinterpret_cast<float32*>(pGradientPixels);
	GradientInitialData.SysMemPitch			= _dims.x()*sizeof(Vector4f);
	GradientInitialData.SysMemSlicePitch	= 0;

	m_Up=Dx11Renderer::Create2DTexture(_pDevice,DXGI_FORMAT_R16G16B16A16_FLOAT, _dims, &GradientInitialData);
	m_UpCorrected=Dx11Renderer::Create2DTexture(_pDevice,DXGI_FORMAT_R16G16B16A16_FLOAT, _dims, &GradientInitialData);

	if (m_bCreateDivergenceFree)
	{
		m_FLIPCorrectedUp=Dx11Renderer::Create2DTexture(_pDevice,DXGI_FORMAT_R16G16B16A16_FLOAT, _dims, &GradientInitialData);	

		D3D11_SUBRESOURCE_DATA ScalarInitialData;
		ScalarInitialData.pSysMem				= reinterpret_cast<float32*>(pGradientPixels);
		ScalarInitialData.SysMemPitch			= _dims.x()*sizeof(Vector1f);
		ScalarInitialData.SysMemSlicePitch		= 0;

		m_Div=Dx11Renderer::Create2DTexture(_pDevice,DXGI_FORMAT_R16_FLOAT, _dims, &ScalarInitialData);	
		m_P[0]=Dx11Renderer::Create2DTexture(_pDevice,DXGI_FORMAT_R16_FLOAT, _dims, &ScalarInitialData);	
		m_P[1]=Dx11Renderer::Create2DTexture(_pDevice,DXGI_FORMAT_R16_FLOAT, _dims, &ScalarInitialData);	
	}
	
	M::DeleteArray(&pGradientPixels);
}

///<
GPUSPH2D::~GPUSPH2D()
{
	for(int32 i=0; i<2;++i)
	{
		M::Release(&m_pParticlesVertexBuffer[i]);
	}

	M::Release(&m_Up);
	M::Release(&m_UpCorrected);

 M::Release(&m_Div);
 M::Release(&m_P[0]);
 M::Release(&m_P[1]);
 M::Release(&m_FLIPCorrectedUp);
 M::Release(&m_pAddPressureGradient);
 M::Release(&m_pComputeDiv);
 M::Release(&m_pJacobi);

	M::Delete(&m_pPostQuad);

	M::Delete(&m_pSplat);
	M::Delete(&m_pAddDensityGradient);
	M::Delete(&m_pRenderParticles);

	///< 
	M::Release(&m_pSPHParticleLayout);
	M::Release(&m_pAdvanceParticlesVS);
	M::Release(&m_pAdvanceParticlesGS);

	M::Release(&m_pConstants);
}