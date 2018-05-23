
#include <Physics/GPUFluids/GPUSPH.hpp>
#include <Physics/GPUFluids/GPUSPHShallowWater.hpp>

#include <Math/Matrix/Matrix.hpp>
#include <Graphics/Camera/Camera.hpp>

#include <Graphics/MeshImport.hpp>
#include <Graphics/Dx11/Mesh.hpp>

///<
void GPUSPHShallowWater::SplatParticles(ID3D11DeviceContext* _pImmediateContext)
{
	m_pSplat->Set(_pImmediateContext);

	_pImmediateContext->OMSetDepthStencilState(List::States::Get().m_pNoDepth, 0);
	_pImmediateContext->PSSetSamplers(0,1,&List::States::Get().m_pPointSampler);
	_pImmediateContext->PSSetSamplers(1,1,&List::States::Get().m_pLinearSampler);
	_pImmediateContext->RSSetState(List::States::Get().m_pNoCullBackRasterizer);	

	float BlackColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f }; 
	_pImmediateContext->OMSetRenderTargets(1, &m_Up._pRTV, NULL);
	_pImmediateContext->ClearRenderTargetView(m_Up._pRTV, BlackColor);
	_pImmediateContext->ClearRenderTargetView(m_UpCorrected._pRTV, BlackColor);

	_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	float32 AddFactor[4]={1.0f, 1.0f, 1.0f, 1.0f};
	_pImmediateContext->OMSetBlendState(List::States::Get().m_pAddBlendState, AddFactor, 0xffffffff);
	
	uint32 offset = 0;
	uint32 stride = sizeof(Particle);
	_pImmediateContext->IASetVertexBuffers(0, 1, &m_pParticlesVertexBuffer[0], &stride, &offset );

	D3D11_VIEWPORT vp; 
	vp.MinDepth = 0.0f;   vp.MaxDepth = 1.0f;  vp.TopLeftX = 0;   vp.TopLeftY = 0;
	vp.Width	= (float32)GetDims().x();  	vp.Height	=(float32) GetDims().y();
	_pImmediateContext->RSSetViewports(1, &vp);

	_pImmediateContext->Draw(m_iNumParticles,0);
	Dx11Renderer::UnbindResources(_pImmediateContext,0,1);
}

///<
void GPUSPHShallowWater::AddDensityGradient(ID3D11DeviceContext* _pImmediateContext)
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
		_pImmediateContext->PSSetShaderResources(2,1,&m_pOceanBottomSRV);

		_pImmediateContext->PSSetShaderResources(3,1,&m_pDensityFieldSRV);

		List::States::Get().SetSamplers(_pImmediateContext);

		_pImmediateContext->OMSetRenderTargets(1, &m_UpCorrected._pRTV, NULL);			
		m_pPostQuad->Draw(_pImmediateContext);
		
		Dx11Renderer::UnbindResources(_pImmediateContext, 0, 4);
	}	
}

///<
void GPUSPHShallowWater::AdvanceParticles(ID3D11DeviceContext* _pImmediateContext, Texture2D_SV& _correctedField)
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
	_pImmediateContext->Draw(m_iNumParticles,0);

	ID3D11Buffer* pBuffers[1]={NULL};
	_pImmediateContext->SOSetTargets(1, pBuffers, &offset);
	std::swap(m_pParticlesVertexBuffer[0],m_pParticlesVertexBuffer[1]);
	Dx11Renderer::UnbindResources(_pImmediateContext, 0, 2);

}
///<
void GPUSPHShallowWater::Update(ID3D11DeviceContext* _pImmediateContext, ID3D11RenderTargetView* _pRT,  ID3D11DepthStencilView* _pDSV, Vector2i _screenDims)
{

	ID3D11Buffer* pBuffer = GPUSPHConstants::Get().GetConstantsBuffer();
	
	_pImmediateContext->GSSetConstantBuffers(3, 1, &pBuffer);
	_pImmediateContext->PSSetConstantBuffers(3, 1, &pBuffer);
	_pImmediateContext->VSSetConstantBuffers(3, 1, &pBuffer);

	///< Draw particle density.
	SplatParticles(_pImmediateContext);	
	
	///< Compute Gradients
	AddDensityGradient(_pImmediateContext);
	
	///< Advect Particles
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
void GPUSPHShallowWater::DrawParticles(ID3D11DeviceContext* _pImmediateContext)
{

	ID3D11Buffer* pConstants=GPUSPHConstants::Get().GetConstantsBuffer();
	_pImmediateContext->GSSetConstantBuffers(3, 1, &pConstants);
	_pImmediateContext->PSSetConstantBuffers(3, 1, &pConstants);
	_pImmediateContext->VSSetConstantBuffers(3, 1, &pConstants);

	m_pRenderParticles->Set(_pImmediateContext);
	_pImmediateContext->IASetInputLayout(m_pSPHParticleLayout);

	_pImmediateContext->RSSetState(List::States::Get().m_pNoCullBackRasterizer);				
	_pImmediateContext->OMSetDepthStencilState(List::States::Get().m_pDepthLess, 0);
	_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	uint32 stride = sizeof(Particle);
	uint32 offset = 0;
	_pImmediateContext->IASetVertexBuffers(0, 1, &m_pParticlesVertexBuffer[0], &stride, &offset);
	_pImmediateContext->Draw(m_iNumParticles,0); //
	_pImmediateContext->GSSetShader(NULL, NULL, 0);
}

///<
void GPUSPHShallowWater::CreateParticles(ID3D11Device* _pDevice)
{

	m_World = Matrix4f::Identity();
	
	m_iNumParticles = m_iParticlesPerAxis.x()*m_iParticlesPerAxis.y();

	D3D11_BUFFER_DESC bd;
	memset(&bd,0,sizeof(D3D11_BUFFER_DESC));

	bd.Usage			= D3D11_USAGE_DEFAULT;
	bd.ByteWidth		= sizeof(Particle)*m_iNumParticles;

	bd.BindFlags		= D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_STREAM_OUTPUT | D3D11_BIND_SHADER_RESOURCE;

	Particle* pVertices = new Particle[m_iNumParticles];
	memset(pVertices,0,sizeof(Particle)*m_iNumParticles);
	
	Vector2f fDims(m_iDims);

	Vector2f particleSpacing		= Vector2f(0.2f)/(Vector2f)m_iParticlesPerAxis;
	Vector2f baseX					= Vector2f(0.4f,0.4f);

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
void GPUSPHShallowWater::Create(ID3D11Device* _pDevice, ID3D11DeviceContext* _pImmediateContext, const char* _strGroundTexture)
{

	std::string baseTexture = "..\\..\\Ressources\\Textures\\";
	std::string tFileName = std::string(_strGroundTexture);
	if (tFileName.length()!=0)
	{
		HRESULT hr = D3DX11CreateShaderResourceViewFromFile(_pDevice, (baseTexture+tFileName).c_str(), NULL, NULL, &m_pOceanBottomSRV, NULL );
		ASSERT(hr ==S_OK, "Failed loading texture !");

		D3D11_TEXTURE2D_DESC texDesc;
		memset(&texDesc,0,sizeof( D3D11_TEXTURE2D_DESC));
		ID3D11Texture2D* pTexture;
		m_pOceanBottomSRV->GetResource((ID3D11Resource**)&pTexture);
		ASSERT(pTexture!=NULL, "Failed getting bottom of ocean");
		if (pTexture)
		{
			pTexture->GetDesc(&texDesc);
			m_iDims = Vector2i(texDesc.Width, texDesc.Height);
			m_iParticlesPerAxis = 0.8f*(Vector2f)m_iDims;
			pTexture->Release();
		}
		else
		{
			m_iDims = Vector2ui(128,128);
			m_iParticlesPerAxis = 100;
		}
	}

	const char* csShallowWaterShaderName = "..\\..\\Src\\Graphics\\Shaders\\Physics\\SPHShallowWater.fx";
	{

		// Define the input layout
		D3D11_INPUT_ELEMENT_DESC particleLayout[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, sizeof(Vector4f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
		};	

		m_pRenderParticles=new Shader();
		Dx11Renderer::CreateShadersAndLayout(csShallowWaterShaderName, "VS_RenderParticles", "PS_RenderParticles", "GS_RenderParticles", particleLayout, ARRAYSIZE(particleLayout), m_pRenderParticles, _pDevice);

		m_pSplat = new Shader();
		Dx11Renderer::CreateShadersAndLayout(csShallowWaterShaderName, "Splatting_VS", "Splatting_PS", "Splatting_GS", particleLayout, ARRAYSIZE(particleLayout), m_pSplat, _pDevice);	

		///< Particle Transport:
		Dx11Renderer::CreateInputLayout(csShallowWaterShaderName,"VS_AdvanceParticles", _pDevice, particleLayout, ARRAYSIZE(particleLayout), &m_pSPHParticleLayout);

		Dx11Renderer::CreateVertexShader(csShallowWaterShaderName,"VS_AdvanceParticles",_pDevice, &m_pAdvanceParticlesVS);

		///< Create Stream Geometry Shader
		{
			ID3DBlob* pVSBlob = NULL;
			HRESULT hr = Dx11Renderer::CompileShaderFromFile(csShallowWaterShaderName, "VS_AdvanceParticles", Dx11Renderer::VSLevel(_pDevice), &pVSBlob);
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
		Dx11Renderer::CreateShadersAndLayout(csShallowWaterShaderName, "RenderQuad_VS", "PS_AddDensityGradient", NULL, postLayout, ARRAYSIZE(postLayout), m_pAddDensityGradient, _pDevice);
	}

	CreateTextures(_pDevice);
	CreateParticles(_pDevice);

	m_pPostQuad = new QuadUV();
	m_pPostQuad->Create(_pDevice, NULL);	

}

///<
void GPUSPHShallowWater::ModifyConstants()
{
	///< Default Values
	GPUSPHConstants::Get().m_constants._Gravity=Vector4f(1,0.0f,0,0);
	GPUSPHConstants::Get().m_constants._InitialDensity[0]=8.5f;
	GPUSPHConstants::Get().m_constants._PressureScale=3.0f;
	GPUSPHConstants::Get().m_constants._PIC_FLIP=0.3f;
	
	GPUSPHConstants::Get().m_constants._TimeStep=0.05f;
}

///<
void GPUSPHShallowWater::CreateMenu()
{
	TwBar* pBar = TwGetBarByIndex(1);

	if (pBar)
	{	
		TwAddVarRO(pBar, "Num Particles", TW_TYPE_INT32, &m_iNumParticles, " label='Num Particles' ");
		TwAddVarRO(pBar, "Grid Dim", TW_TYPE_INT32, &m_iDims[0], " label='Grid Dims' ");
	}
}

///<
void GPUSPHShallowWater::CreateTextures(ID3D11Device* _pDevice)
{
	
	Vector4f* pGradientPixels = new Vector4f[m_iDims.x()*m_iDims.y()];
	memset(pGradientPixels, 0, sizeof(Vector4f)*m_iDims.x()*m_iDims.y());

	FOR_EACH_VECTOR2_COMPONENT(m_iDims)
	{
		pGradientPixels[m_iDims.x()*j + i]=Vector4f(0,0,0,0);
	}
	END_FOR_EACH
	

	///< Check first 3D.
	D3D11_SUBRESOURCE_DATA GradientInitialData;
	GradientInitialData.pSysMem				= reinterpret_cast<float32*>(pGradientPixels);
	GradientInitialData.SysMemPitch			= m_iDims.x()*sizeof(Vector4f);
	GradientInitialData.SysMemSlicePitch	= 0;

	m_Up=Dx11Renderer::Create2DTexture(_pDevice,DXGI_FORMAT_R32G32B32A32_FLOAT, m_iDims, &GradientInitialData);
	m_UpCorrected=Dx11Renderer::Create2DTexture(_pDevice,DXGI_FORMAT_R32G32B32A32_FLOAT, m_iDims, &GradientInitialData);	
	
	M::DeleteArray(&pGradientPixels);
}

///<
GPUSPHShallowWater::~GPUSPHShallowWater()
{
	for(int32 i=0; i<2;++i)
	{
		M::Release(&m_pParticlesVertexBuffer[i]);
	}

	M::Release(&m_Up);
	M::Release(&m_UpCorrected);

	M::Delete(&m_pPostQuad);

	M::Delete(&m_pSplat);
	M::Delete(&m_pAddDensityGradient);
	M::Delete(&m_pRenderParticles);

	M::Release(&m_pOceanBottomSRV);

	///< 
	M::Release(&m_pSPHParticleLayout);
	M::Release(&m_pAdvanceParticlesVS);
	M::Release(&m_pAdvanceParticlesGS);

}