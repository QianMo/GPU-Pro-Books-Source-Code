
#include <Graphics/Dx11/Post/FXAA.hpp>
#include <Input/Keyboard.hpp>

///<
void FXAA::CreateMenu()
{

	TwBar* pBar = TwGetBarByIndex(1);

	if (pBar)
	{
		TwAddVarRW(pBar, "FXAA", TW_TYPE_BOOLCPP, &m_bFXAA, "label='FXAA?'");
	}
}

///<
void FXAA::Create(ID3D11Device* _pDevice, const DXGI_SWAP_CHAIN_DESC* pSwapChainDesc)
{
	m_bFXAA = true;

	ASSERT(pSwapChainDesc!=NULL, "NULL ptr!");

	D3D11_TEXTURE2D_DESC desc;
	memset(&desc, 0, sizeof(desc));
	desc.BindFlags			= D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	desc.Format				= DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.Height				= pSwapChainDesc->BufferDesc.Height;
	desc.Width				= pSwapChainDesc->BufferDesc.Width;
	desc.ArraySize			= 1;
	desc.SampleDesc.Count	= pSwapChainDesc->SampleDesc.Count;
	desc.SampleDesc.Quality = pSwapChainDesc->SampleDesc.Quality;
	desc.MipLevels			= 1;

	_pDevice->CreateTexture2D(&desc, 0, &m_pRenderTargetTexture);
	_pDevice->CreateRenderTargetView(m_pRenderTargetTexture, 0, &m_pRenderTarget);
	_pDevice->CreateShaderResourceView(m_pRenderTargetTexture, 0, &m_pRenderTargetTextureSRV);

	CreateShaders(_pDevice);

	m_pQuadVertexBuffer = Dx11Renderer::CreatePostProcessQuad(_pDevice);

	CreateConstantBuffers(_pDevice);

	CreateMenu();
}

void FXAA::Render(ID3D11DeviceContext* _pImmediateContext)
{
	m_aaParams.AA_LEVEL=Vector4f(m_bFXAA,0,0,0);

	_pImmediateContext->UpdateSubresource(m_pAAParams, 0, NULL, &m_aaParams, 0, 0);

	uint32 stride = sizeof(Vector3f);
	uint32 offset = 0;
	_pImmediateContext->IASetVertexBuffers(0, 1, &m_pQuadVertexBuffer, &stride, &offset);	
	_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	m_pShader->Set(_pImmediateContext);
	_pImmediateContext->PSSetConstantBuffers(0, 1, &m_pAAParams);	

	_pImmediateContext->PSSetShaderResources(0,1,&m_pRenderTargetTextureSRV);
	_pImmediateContext->PSSetSamplers(0,1,&m_pSamAni);

	_pImmediateContext->Draw(6, 0);	

	// Unbind render targets
	_pImmediateContext->OMSetRenderTargets(0, 0, 0); 

	// Unbind shader resources
	ID3D11ShaderResourceView* nullViews[1] = {0};
	_pImmediateContext->PSSetShaderResources(0, 1, nullViews);
}


///<
void FXAA::CreateShaders(ID3D11Device* _pDevice)
{
	ID3DBlob* pBlob = NULL;

	const char* csShaderName="..\\..\\Src\\Graphics\\Shaders\\FXAA.fx";
	m_pShader = new Shader();
	///<
	D3D11_INPUT_ELEMENT_DESC layout[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};
	UINT numElements = ARRAYSIZE(layout);

	Dx11Renderer::CreateShadersAndLayout(csShaderName, "FXAA_VS", "FXAA_PS", NULL, layout, ARRAYSIZE(layout), m_pShader, _pDevice);	

	///< Create Sampler
	{
		D3D11_SAMPLER_DESC samDesc;
		memset(&samDesc, 0, sizeof(D3D11_SAMPLER_DESC));

		samDesc.Filter = D3D11_FILTER_ANISOTROPIC;
		samDesc.AddressU = samDesc.AddressV = samDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		samDesc.MaxAnisotropy = 4;
		samDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
		samDesc.MaxLOD = 0.0f;
		samDesc.MinLOD = 0.0f;
		_pDevice->CreateSamplerState(&samDesc, &m_pSamAni);
	}
}


///<
void FXAA::Release()
{
	M::Delete(&m_pShader);

	M::Release(&m_pQuadVertexBuffer);   
	M::Release(&m_pSamAni);   

	M::Release(&m_pRenderTargetTexture);
	M::Release(&m_pRenderTargetTextureSRV);
	M::Release(&m_pRenderTarget);	

}

///<
void FXAA::CreateConstantBuffers(ID3D11Device* _pDevice)
{
	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));

	// Create the constant buffers
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(CBAAParams);
	bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	bd.CPUAccessFlags = 0;
	HRESULT hr = _pDevice->CreateBuffer(&bd, NULL, &m_pAAParams );
	ASSERT(hr==S_OK, "Failed To Creat Constant Buffer");
}