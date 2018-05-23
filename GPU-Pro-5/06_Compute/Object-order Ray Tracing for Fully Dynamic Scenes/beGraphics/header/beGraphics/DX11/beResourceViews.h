/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_RESOURCEVIEWS_DX11
#define BE_GRAPHICS_RESOURCEVIEWS_DX11

#include "beGraphics.h"
#include <beCore/beWrapper.h>
#include <D3D11.h>
#include <lean/smart/com_ptr.h>

namespace beGraphics
{

namespace DX11
{

/// Resource view wrapper.
template <class ID3D11ViewInterface, class D3D11ViewDesc>
class ResourceView : public beCore::IntransitiveWrapper< ID3D11ViewInterface, ResourceView<ID3D11ViewInterface, D3D11ViewDesc> >
{
private:
	lean::com_ptr<ID3D11ViewInterface> m_pView;

public:
	/// View description type.
	typedef D3D11ViewDesc ViewDesc;
	/// View type.
	typedef ID3D11ViewInterface ViewType; 

	/// Constructor.
	BE_GRAPHICS_DX11_API ResourceView(ID3D11Resource *pResource, const ViewDesc &desc);
	/// Constructor.
	BE_GRAPHICS_DX11_API ResourceView(ViewType *pView);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~ResourceView();

	/// Gets the D3D resource view.
	LEAN_INLINE ViewType*const& GetInterface() const { return m_pView.get(); }
	/// Gets the D3D resource view.
	LEAN_INLINE ViewType*const& GetView() const { return m_pView.get(); }
};

/// Shader resource view wrapper.
typedef ResourceView<ID3D11ShaderResourceView, D3D11_SHADER_RESOURCE_VIEW_DESC> ShaderResourceView;
/// Render target view wrapper.
typedef ResourceView<ID3D11RenderTargetView, D3D11_RENDER_TARGET_VIEW_DESC> RenderTargetView;
/// Depth-stencil view wrapper.
typedef ResourceView<ID3D11DepthStencilView, D3D11_DEPTH_STENCIL_VIEW_DESC> DepthStencilView;

#ifdef BE_GRAPHICS_RESOURCEVIEWS_DX11_INSTANTIATE
	template class ResourceView<ID3D11ShaderResourceView, D3D11_SHADER_RESOURCE_VIEW_DESC>;
	template class ResourceView<ID3D11RenderTargetView, D3D11_RENDER_TARGET_VIEW_DESC>;
	template class ResourceView<ID3D11DepthStencilView, D3D11_DEPTH_STENCIL_VIEW_DESC>;
#endif

} // namespace

} // namespace

#endif