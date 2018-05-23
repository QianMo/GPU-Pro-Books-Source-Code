/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_STATE_DX11
#define BE_GRAPHICS_STATE_DX11

#include "beGraphics.h"
#include <beCore/beWrapper.h>
#include <D3D11.h>
#include <lean/smart/com_ptr.h>

namespace beGraphics
{

namespace DX11
{

/// State wrapper.
template <class ID3D11StateInterface, class D3D11StateDesc>
class State : public beCore::IntransitiveWrapper< ID3D11StateInterface, State<ID3D11StateInterface, D3D11StateDesc> >
{
private:
	lean::com_ptr<ID3D11StateInterface> m_pState;

public:
	/// State description type.
	typedef D3D11StateDesc StateDesc;
	/// State type.
	typedef ID3D11StateInterface StateType; 

	/// Constructor.
	BE_GRAPHICS_DX11_API State(const StateDesc &desc, ID3D11Device *pDevice);
	/// Constructor.
	BE_GRAPHICS_DX11_API State(StateType *pState);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~State();

	/// Gets the D3D state block.
	LEAN_INLINE StateType*const& GetInterface() const { return m_pState.get(); }
	/// Gets the D3D state block.
	LEAN_INLINE StateType*const& GetState() const { return m_pState.get(); }
};

/// Rasterizer state wrapper.
typedef State<ID3D11RasterizerState, D3D11_RASTERIZER_DESC> RasterizerStateDX11;
/// Depth-stencil state state wrapper.
typedef State<ID3D11DepthStencilState, D3D11_DEPTH_STENCIL_DESC> DepthStencilStateDX11;
/// Blend state state wrapper.
typedef State<ID3D11BlendState, D3D11_BLEND_DESC> BlendStateDX11;

#ifdef BE_GRAPHICS_STATE_DX11_INSTANTIATE
	template class State<ID3D11RasterizerState, D3D11_RASTERIZER_DESC>;
	template class State<ID3D11DepthStencilState, D3D11_DEPTH_STENCIL_DESC>;
	template class State<ID3D11BlendState, D3D11_BLEND_DESC>;
#endif

} // namespace

} // namespace

#endif