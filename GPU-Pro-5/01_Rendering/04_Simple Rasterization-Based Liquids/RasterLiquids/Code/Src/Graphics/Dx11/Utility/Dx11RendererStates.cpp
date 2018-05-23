
#include <Graphics/Dx11/Utility/Dx11RendererStates.hpp>


//////////////////////////////////////////////////////////////////////////
///< Rasterizer states and other states
//////////////////////////////////////////////////////////////////////////

List::States List::States::m_inst;

///<
void List::States::SetSamplers(ID3D11DeviceContext* _pImmediateContext)
{
	_pImmediateContext->PSSetSamplers(0,1,&m_pPointSampler);
	_pImmediateContext->PSSetSamplers(1,1,&m_pLinearSampler);
}

///<
void List::States::Release()
{
	M::Release(&m_pCullFrontRasterizer);
	M::Release(&m_pCullBackRasterizer);
	M::Release(&m_pNoCullBackRasterizer);
	M::Release(&m_pNoCullBackWireframeRasterizer);

	M::Release(&m_pBlendingRasterizer);
	M::Release(&m_pLinearSampler);
	M::Release(&m_pPointSampler);		

	M::Release(&m_pDepthLess);
	M::Release(&m_pNoDepth);

	M::Release(&m_pDefaultNoBlendState);
	M::Release(&m_pAddBlendState);
	
	M::Release(&m_pTransparentBlendState);
}

void List::States::Create(ID3D11Device* _pDevice)
{
	///< Rasterizers
	{
		///< 
		D3D11_RASTERIZER_DESC rastDesc;
		memset(&rastDesc, 0, sizeof(rastDesc));
		rastDesc.CullMode = D3D11_CULL_BACK;
		rastDesc.FillMode = D3D11_FILL_SOLID;
		rastDesc.AntialiasedLineEnable = FALSE;
		rastDesc.DepthBias = 0;
		rastDesc.DepthBiasClamp = 0;
		rastDesc.DepthClipEnable = FALSE;
		rastDesc.FrontCounterClockwise = FALSE;
		rastDesc.MultisampleEnable = FALSE;
		rastDesc.ScissorEnable = FALSE;
		rastDesc.SlopeScaledDepthBias = 0;

		HRESULT hr = _pDevice->CreateRasterizerState(&rastDesc, &m_pCullBackRasterizer);
		ASSERT(hr==S_OK, "Failed!");

		///< CCW
		{
			rastDesc.CullMode = D3D11_CULL_FRONT;
			
			hr = _pDevice->CreateRasterizerState(&rastDesc, &m_pCullFrontRasterizer);
			ASSERT(hr==S_OK, "Failed!");			
			rastDesc.CullMode = D3D11_CULL_BACK;
		}

		///< For Blending!
		{
			rastDesc.DepthClipEnable = FALSE;
			HRESULT hr = _pDevice->CreateRasterizerState(&rastDesc, &m_pBlendingRasterizer);
			rastDesc.DepthClipEnable = TRUE;

			ASSERT(hr==S_OK, "Failed!");
		}

		///< Wireframe!
		{
			rastDesc.FillMode = D3D11_FILL_WIREFRAME;
			rastDesc.CullMode = D3D11_CULL_NONE;

			hr = _pDevice->CreateRasterizerState(&rastDesc, &m_pNoCullBackWireframeRasterizer);
			rastDesc.FillMode = D3D11_FILL_SOLID;
			ASSERT(hr==S_OK, "Failed!");
		}

		///< No culling!
		{
			rastDesc.CullMode = D3D11_CULL_NONE;			
			hr = _pDevice->CreateRasterizerState(&rastDesc, &m_pNoCullBackRasterizer);
			ASSERT(hr==S_OK, "Failed!");
		}
	}


	///< No Blending for 1 Render Target
	{
		D3D11_RENDER_TARGET_BLEND_DESC rtDesc;
		rtDesc.BlendEnable		= FALSE;
		rtDesc.BlendOp			= D3D11_BLEND_OP_ADD;
		rtDesc.BlendOpAlpha		= D3D11_BLEND_OP_ADD;
		rtDesc.SrcBlend			= D3D11_BLEND_SRC_COLOR;
		rtDesc.DestBlend		= D3D11_BLEND_DEST_COLOR;
		rtDesc.SrcBlendAlpha	= D3D11_BLEND_SRC_ALPHA;
		rtDesc.DestBlendAlpha	= D3D11_BLEND_DEST_ALPHA;
		rtDesc.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

		///< ...		
		D3D11_BLEND_DESC blendDesc;
		blendDesc.RenderTarget[0] = rtDesc;
		blendDesc.AlphaToCoverageEnable = FALSE;
		blendDesc.IndependentBlendEnable = FALSE;

		HRESULT hr = _pDevice->CreateBlendState(&blendDesc, &m_pDefaultNoBlendState);
		ASSERT(hr==S_OK, "Failed creating blend state");


		///< Add Blend State
		rtDesc.BlendEnable				= TRUE;
		rtDesc.BlendOp					= D3D11_BLEND_OP_ADD;
		rtDesc.SrcBlend					= D3D11_BLEND_ONE;
		rtDesc.DestBlend				= D3D11_BLEND_ONE;

		rtDesc.BlendOpAlpha				= D3D11_BLEND_OP_ADD;
		rtDesc.SrcBlendAlpha			= D3D11_BLEND_ONE;			
		rtDesc.DestBlendAlpha			= D3D11_BLEND_ONE;			
		rtDesc.RenderTargetWriteMask	= D3D11_COLOR_WRITE_ENABLE_ALL;

		///< ...		
		blendDesc.RenderTarget[0] = rtDesc;
		blendDesc.AlphaToCoverageEnable = FALSE;
		blendDesc.IndependentBlendEnable = FALSE;

		///< Create
		hr = _pDevice->CreateBlendState(&blendDesc, &m_pAddBlendState);
		ASSERT(hr==S_OK, "Failed!");


		///< Transparent Blend State
		rtDesc.BlendEnable				= TRUE;
		rtDesc.BlendOp					= D3D11_BLEND_OP_ADD;
		rtDesc.SrcBlend					= D3D11_BLEND_SRC_ALPHA;
		rtDesc.DestBlend				= D3D11_BLEND_INV_SRC_ALPHA;

		rtDesc.BlendOpAlpha				= D3D11_BLEND_OP_ADD;
		rtDesc.SrcBlendAlpha			= D3D11_BLEND_SRC_ALPHA;			
		rtDesc.DestBlendAlpha			= D3D11_BLEND_INV_SRC_ALPHA;			
		rtDesc.RenderTargetWriteMask	= D3D11_COLOR_WRITE_ENABLE_ALL;

		///< ...		
		blendDesc.RenderTarget[0] = rtDesc;
		blendDesc.AlphaToCoverageEnable = FALSE;
		blendDesc.IndependentBlendEnable = FALSE;

		///< Create
		hr = _pDevice->CreateBlendState(&blendDesc, &m_pTransparentBlendState);
		ASSERT(hr==S_OK, "Failed!");

	}

	///< Depth Test
	{
		D3D11_DEPTH_STENCIL_DESC DepthDesc;

		DepthDesc.DepthEnable = TRUE;
		DepthDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
		DepthDesc.DepthFunc = D3D11_COMPARISON_LESS;
		DepthDesc.StencilEnable = FALSE;
		_pDevice->CreateDepthStencilState(&DepthDesc, &m_pDepthLess);		

		///<
		DepthDesc.DepthEnable = FALSE;
		DepthDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
		DepthDesc.DepthFunc = D3D11_COMPARISON_NEVER;
		_pDevice->CreateDepthStencilState(&DepthDesc, &m_pNoDepth);

	}

	///< Create Sampler
	{
		D3D11_SAMPLER_DESC samDesc;
		memset(&samDesc, 0, sizeof(samDesc));

		samDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		samDesc.AddressU = samDesc.AddressV = samDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		samDesc.MaxAnisotropy = 1;
		samDesc.MipLODBias=0;
		samDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
		samDesc.MaxLOD = D3D11_FLOAT32_MAX;
		samDesc.MinLOD = 0;
		HRESULT hr = _pDevice->CreateSamplerState(&samDesc, &m_pLinearSampler);
		ASSERT(hr==S_OK, "Failed create sampler.");
	}

	{
		D3D11_SAMPLER_DESC samDesc;
		memset(&samDesc, 0, sizeof(D3D11_SAMPLER_DESC));

		samDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
		samDesc.AddressU = samDesc.AddressV = samDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		samDesc.MaxAnisotropy = 0;
		samDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
		samDesc.MaxLOD = 0.0f;
		samDesc.MinLOD = 0.0f;
		_pDevice->CreateSamplerState(&samDesc, &m_pPointSampler);
	}
}