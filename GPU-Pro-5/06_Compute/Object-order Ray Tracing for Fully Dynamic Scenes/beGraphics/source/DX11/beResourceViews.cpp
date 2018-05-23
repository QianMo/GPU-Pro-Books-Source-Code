/*****************************************************/
/* breeze Engine Render Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"

#define BE_GRAPHICS_RESOURCEVIEWS_DX11_INSTANTIATE

#include "beGraphics/DX11/beResourceViews.h"
#include "beGraphics/DX/beError.h"

namespace beGraphics
{

namespace DX11
{

namespace
{

/// Creates a shader resource view
lean::com_ptr<ID3D11ShaderResourceView, true> CreateResourceView(ID3D11Resource *pResource, const D3D11_SHADER_RESOURCE_VIEW_DESC& desc)
{
	lean::com_ptr<ID3D11Device> pDevice;
	pResource->GetDevice(pDevice.rebind());

	lean::com_ptr<ID3D11ShaderResourceView> pView;

	BE_THROW_DX_ERROR_MSG(
		pDevice->CreateShaderResourceView(pResource, &desc, pView.rebind()),
		"ID3D11Device::CreateShaderResourceView()");

	return pView.transfer();
}

/// Creates a render target resource view
lean::com_ptr<ID3D11RenderTargetView, true> CreateResourceView(ID3D11Resource *pResource, const D3D11_RENDER_TARGET_VIEW_DESC& desc)
{
	lean::com_ptr<ID3D11Device> pDevice;
	pResource->GetDevice(pDevice.rebind());

	lean::com_ptr<ID3D11RenderTargetView> pView;

	BE_THROW_DX_ERROR_MSG(
		pDevice->CreateRenderTargetView(pResource, &desc, pView.rebind()),
		"ID3D11Device::CreateRenderTargetView()");

	return pView.transfer();
}

/// Creates a depth-stencil resource view
lean::com_ptr<ID3D11DepthStencilView, true> CreateResourceView(ID3D11Resource *pResource, const D3D11_DEPTH_STENCIL_VIEW_DESC& desc)
{
	lean::com_ptr<ID3D11Device> pDevice;
	pResource->GetDevice(pDevice.rebind());

	lean::com_ptr<ID3D11DepthStencilView> pView;

	BE_THROW_DX_ERROR_MSG(
		pDevice->CreateDepthStencilView(pResource, &desc, pView.rebind()),
		"ID3D11Device::CreateDepthStencilView()");

	return pView.transfer();
}

} // namespace

// Constructor.
template <class ID3D11ViewInterface, class D3D11ViewDesc>
ResourceView<ID3D11ViewInterface, D3D11ViewDesc>::ResourceView(ID3D11Resource *pResource, const ViewDesc& desc)
	: m_pView( CreateResourceView(pResource, desc) )
{
}

// Constructor.
template <class ID3D11ViewInterface, class D3D11ViewDesc>
ResourceView<ID3D11ViewInterface, D3D11ViewDesc>::ResourceView(ID3D11ViewInterface *pView)
	: m_pView(pView)
{
	LEAN_ASSERT(m_pView != nullptr);
}

// Destructor.
template <class ID3D11ViewInterface, class D3D11ViewDesc>
ResourceView<ID3D11ViewInterface, D3D11ViewDesc>::~ResourceView()
{
}

} // namespace

} // namespace
