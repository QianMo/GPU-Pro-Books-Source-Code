/*****************************************************/
/* breeze Engine Render Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"

#define BE_GRAPHICS_STATE_DX11_INSTANTIATE

#include "beGraphics/DX11/beState.h"
#include "beGraphics/DX/beError.h"

namespace beGraphics
{

namespace DX11
{

namespace
{

/// Creates a rasterizer state block.
lean::com_ptr<ID3D11RasterizerState, true> CreateState(const D3D11_RASTERIZER_DESC &desc, ID3D11Device *pDevice)
{
	lean::com_ptr<ID3D11RasterizerState> pState;
	
	BE_THROW_DX_ERROR_MSG(
		pDevice->CreateRasterizerState(&desc, pState.rebind()),
		"ID3D11Device::CreateRasterizerState()" );

	return pState.transfer();
}

/// Creates a depth-stencil state block.
lean::com_ptr<ID3D11DepthStencilState, true> CreateState(const D3D11_DEPTH_STENCIL_DESC &desc, ID3D11Device *pDevice)
{
	lean::com_ptr<ID3D11DepthStencilState> pState;
	
	BE_THROW_DX_ERROR_MSG(
		pDevice->CreateDepthStencilState(&desc, pState.rebind()),
		"ID3D11Device::CreateDepthStencilState()" );

	return pState.transfer();
}

/// Creates a blend state block.
lean::com_ptr<ID3D11BlendState, true> CreateState(const D3D11_BLEND_DESC &desc, ID3D11Device *pDevice)
{
	lean::com_ptr<ID3D11BlendState> pState;
	
	BE_THROW_DX_ERROR_MSG(
		pDevice->CreateBlendState(&desc, pState.rebind()),
		"ID3D11Device::CreateBlendState()" );

	return pState.transfer();
}

} // namespace

// Constructor.
template <class ID3D11StateInterface, class D3D11StateDesc>
State<ID3D11StateInterface, D3D11StateDesc>::State(const StateDesc &desc, ID3D11Device *pDevice)
	: m_pState( CreateState(desc, pDevice) )
{
}

// Constructor.
template <class ID3D11StateInterface, class D3D11StateDesc>
State<ID3D11StateInterface, D3D11StateDesc>::State(ID3D11StateInterface *pState)
	: m_pState(pState)
{
	LEAN_ASSERT(m_pState != nullptr);
}

// Destructor.
template <class ID3D11StateInterface, class D3D11StateDesc>
State<ID3D11StateInterface, D3D11StateDesc>::~State()
{
}

} // namespace

} // namespace