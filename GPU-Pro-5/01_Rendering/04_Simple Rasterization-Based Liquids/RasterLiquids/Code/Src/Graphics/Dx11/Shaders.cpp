
#include <Graphics/Dx11/Shaders.hpp>
#include <Graphics/Dx11/Mesh.hpp>


#include <Common/Common.hpp>
#include <Input\MenuManager.hpp>

void Shader::Release()
{
	M::Release(&m_pVertex);
	M::Release(&m_pPixel);
	M::Release(&m_pLayout);
	M::Release(&m_pGeometry);
}

///<
void Shader::Set(ID3D11DeviceContext* _pContext)
{
	if (m_pLayout)
		_pContext->IASetInputLayout(m_pLayout);	

	if (m_pVertex)
		_pContext->VSSetShader(m_pVertex, NULL, 0);
	else
		_pContext->VSSetShader(NULL, NULL, 0);

	if (m_pGeometry)
		_pContext->GSSetShader(m_pGeometry, NULL, 0);	
	else
		_pContext->GSSetShader(NULL, NULL, 0);

	if(m_pPixel)
		_pContext->PSSetShader(m_pPixel, NULL, 0);
	else
		_pContext->PSSetShader(NULL, NULL, 0);

}

///<
RayCastShader::~RayCastShader()
{
	M::Release(&m_pVertex);
	M::Release(&m_pLayout);
	M::Release(&m_pDrawBackFaces);
	M::Release(&m_pDrawFrontFaces);
	M::Release(&m_pRayCastVolume);

	M::Delete(&m_pCube);
	M::Release(&m_RayCastTexture);	
}

///<
void RayCastShader::Draw(ID3D11DeviceContext* _pContext, ID3D11ShaderResourceView* _pVolumeTexture, ID3D11ShaderResourceView* _pDepth, ID3D11RenderTargetView* _pDrawTo, ID3D11ShaderResourceView* _pEnvMap)
{
	List::States::Get().SetSamplers(_pContext);
	_pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	_pContext->OMSetRenderTargets(1, &m_RayCastTexture._pRTV, NULL);
	float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f }; 
	_pContext->ClearRenderTargetView(m_RayCastTexture._pRTV, ClearColor);

	_pContext->RSSetState(List::States::Get().m_pCullFrontRasterizer);	
	_pContext->OMSetBlendState(List::States::Get().m_pDefaultNoBlendState, NULL, 0xffffffff);
	_pContext->IASetInputLayout(m_pLayout);	
	_pContext->VSSetShader(m_pVertex, NULL, 0);
	_pContext->PSSetShader(m_pDrawBackFaces, NULL, 0);
	m_pCube->Draw(_pContext);
	/*
	float32 AddFactor[4]={1.0f, 1.0f, 1.0f, 1.0f};
	_pContext->OMSetBlendState(List::States::Get().m_pAddBlendState, AddFactor, 0xffffffff);
	_pContext->RSSetState(List::States::Get().m_pCullBackRasterizer);
	_pContext->PSSetShader(m_pDrawFrontFaces, NULL, 0);		
	m_pCube->Draw(_pContext);
*/
	_pContext->RSSetState(List::States::Get().m_pCullBackRasterizer);

	Dx11Renderer::UnbindResources(_pContext,0, 1);
	///< Raycast at the point X in the direction of texture.

	float32 AddFactor[4]={1.0f, 1.0f, 1.0f, 1.0f};
	_pContext->OMSetBlendState(List::States::Get().m_pTransparentBlendState, AddFactor, 0xffffffff);
	_pContext->OMSetRenderTargets(1, &_pDrawTo, NULL);				
	_pContext->PSSetShaderResources(0,1, &m_RayCastTexture._pSRV);
	_pContext->PSSetShaderResources(1,1,&_pVolumeTexture);
	_pContext->PSSetShaderResources(2,1,&_pDepth);

	if (_pEnvMap)
		_pContext->PSSetShaderResources(3,1,&_pEnvMap);	
	
	_pContext->OMSetDepthStencilState(List::States::Get().m_pNoDepth, 0);
	_pContext->PSSetShader(m_pRayCastVolume,NULL,0);
	m_pCube->Draw(_pContext);
	Dx11Renderer::UnbindResources(_pContext,0, 4);	
}

///<
void RayCastShader::Create(const char* _csFileName, ID3D11Device* _pDevice, int32 _w, int32 _h)
{
	m_pCube =  new CubeUV();
	m_pCube->Create(_pDevice);
	Matrix4f translation, scale;
	M::AffineTranslation(Vector3f(0.0f,0,0), translation);
	M::AffineScale(Vector3f(10.0f), scale);
	m_pCube->m_World = translation*scale;
	
	m_RayCastTexture = Dx11Renderer::Create2DTexture(_pDevice,DXGI_FORMAT_R32G32B32A32_FLOAT,Vector2i(_w,_h),NULL);

	D3D11_INPUT_ELEMENT_DESC layout[] ={
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, sizeof(Vector3f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};		

	Dx11Renderer::CreateInputLayout(_csFileName, "VS_DrawPosition", _pDevice, layout, ARRAYSIZE(layout), &m_pLayout);
	Dx11Renderer::CreateVertexShader(_csFileName, "VS_DrawPosition", _pDevice, &m_pVertex);

	Dx11Renderer::CreatePixelShader(_csFileName, "PS_DrawBackFaces", _pDevice, &m_pDrawBackFaces);
	Dx11Renderer::CreatePixelShader(_csFileName, "PS_DrawFrontFaces", _pDevice, &m_pDrawFrontFaces);
	Dx11Renderer::CreatePixelShader(_csFileName, "PS_RayCastVolume", _pDevice, &m_pRayCastVolume);
}

void PhongShader::Release()
{
	M::Delete(&m_pPhongShader);
	M::Delete(&m_pDisplacementShader);
}

/////< Phong
PhongShader::~PhongShader()
{
	Release();
}

///<
void PhongShader::Create(ID3D11Device* _pDevice)
{
	///< Layout and Shaders
	{
		const char* csShaderName = "..\\..\\Src\\Graphics\\Shaders\\Phong.fx";

		{

				m_pPhongShader = new Shader();

			D3D11_INPUT_ELEMENT_DESC layout[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, sizeof(Vector3f), D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 2*sizeof(Vector3f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
			};	

			Dx11Renderer::CreateShadersAndLayout(csShaderName, "PhongVS", "PhongPS", NULL, layout, ARRAYSIZE(layout), m_pPhongShader, _pDevice);	
		}		

		{
			m_pDisplacementShader = new Shader();

			D3D11_INPUT_ELEMENT_DESC layoutDisplacement[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, sizeof(Vector3f), D3D11_INPUT_PER_VERTEX_DATA, 0 }
			};

			Dx11Renderer::CreateShadersAndLayout(csShaderName, "PhongDisplacementVS", "PhongPS", NULL, layoutDisplacement, ARRAYSIZE(layoutDisplacement), m_pDisplacementShader, _pDevice);
		}		
	}

	m_lightOrientations[0] = Quaternionf::GenRotation(-0.3f*M::Pi, Vector3f(0,0,1))*Quaternionf::GenRotation(-0.3f*M::Pi, Vector3f(1,0,0)); 
	m_lightOrientations[1] = Quaternionf::GenRotation(-0.3f*M::Pi, Vector3f(0,0,1))*Quaternionf::GenRotation(0.3f*M::Pi, Vector3f(1,0,0)); 

	m_DefaultColor.m_t._c = Vector4f(0.4f,0.55f,0.9f,1);

	for(uint32 i=0; i<m_lightOrientations.Size(); ++i)
		m_LightConstants.m_t._LightPos[i] =  m_lightOrientations[i].Rotate(Vector3f(0,15,0)); 

	///< Constants
	m_DefaultColor.Create(_pDevice, 4);
	m_LightConstants.Create(_pDevice, 2);
}

///<
void PhongShader::UpdateConstants(ID3D11DeviceContext* _pContext)
{

	for(uint32 i=0; i<m_lightOrientations.Size(); ++i)
		m_LightConstants.m_t._LightPos[i] =  m_lightOrientations[i].Rotate(Vector3f(0,15,0)); 

	m_LightConstants.Update(_pContext);

	m_DefaultColor.Update(_pContext);

}




///<
void PhongShader::Set(ID3D11DeviceContext* _pContext)
{
	m_pPhongShader->Set(_pContext);

	UpdateConstants(_pContext);
}

///<
void PhongShader::SetDisplacement(ID3D11DeviceContext* _pContext)
{
	m_pDisplacementShader->Set(_pContext);

	UpdateConstants(_pContext);
}

void PhongShader::CreateMenu()
{
	TwBar* pBar = MenuManager::Get().AddBar("Phong", 3);
	if (pBar)
	{		
		TwAddVarRW(pBar, "LightOrientation1", TW_TYPE_QUAT4F, &m_lightOrientations[0], "label='Light 1 Orientation'");
		TwAddVarRW(pBar, "LightOrientation2", TW_TYPE_QUAT4F, &m_lightOrientations[1], "label='Light 2 Orientation'");
	}
}


ShaderManager	ShaderManager::m_instance;

///<
void ShaderManager::Create(ID3D11Device* _pDevice)
{
	m_phongShader.Create(_pDevice);
}

///<
void ShaderManager::Release()
{
	m_phongShader.Release();
}