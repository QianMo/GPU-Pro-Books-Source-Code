/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_DEVICE_DX11
#define BE_GRAPHICS_DEVICE_DX11

#include "beGraphics.h"
#include "../beDevice.h"
#include <beCore/beWrapper.h>
#include <lean/smart/com_ptr.h>
#include <D3D11.h>
#include <DXGI.h>
#include <vector>

namespace beGraphics
{

namespace DX11
{

/// Gets the device from the given device child.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11Device> GetDevice(ID3D11DeviceChild &deviceChild);

/// Gets the back buffer.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11Texture2D, true> GetBackBuffer(IDXGISwapChain *pSwapChain, uint4 index = 0);

/// Swap chain implementation.
class SwapChain : public beCore::IntransitiveWrapper<IDXGISwapChain, SwapChain>, public beGraphics::SwapChain
{
private:
	lean::com_ptr<IDXGISwapChain> m_pSwapChain;
	
	SwapChainDesc m_desc;
	
public:
	/// Constructor.
	BE_GRAPHICS_DX11_API SwapChain(ID3D11Device *pDevice, const SwapChainDesc &desc, IDXGIOutput *pOutput = nullptr);
	/// Constructor.
	BE_GRAPHICS_DX11_API SwapChain(IDXGISwapChain *pSwapChain);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~SwapChain();

	/// Presents the rendered image.
	BE_GRAPHICS_DX11_API void Present(bool bVSync = false);
	
	/// Resizes the swap chain buffers.
	BE_GRAPHICS_DX11_API void Resize(uint4 width, uint4 height);

	/// Gets the swap chain description.
	BE_GRAPHICS_DX11_API SwapChainDesc GetDesc() const;
	
	/// Gets the D3D swap chain.
	LEAN_INLINE IDXGISwapChain*const& GetInterface() const { return m_pSwapChain.get(); }
	/// Gets the D3D swap chain.
	LEAN_INLINE IDXGISwapChain*const& GetSwapChain() const { return m_pSwapChain.get(); }

	/// Gets the implementation identifier.
	LEAN_INLINE ImplementationID GetImplementationID() const { return DX11Implementation; }
};

template <> struct ToImplementationDX11<beGraphics::SwapChain> { typedef SwapChain Type; };

/// Device implementation.
class Device : public beCore::IntransitiveWrapper<ID3D11Device, Device>, public beGraphics::Device
{
private:
	lean::com_ptr<ID3D11Device> m_pDevice;

	/// Head.
	struct Head
	{
		lean::resource_ptr<SwapChain> pSwapChain;
		Viewport virtualViewport;

		Head(const lean::resource_ptr<SwapChain> &pSwapChain, const Viewport &virtualViewport)
			: pSwapChain(pSwapChain),
			virtualViewport(virtualViewport) { }
	};

	typedef std::vector<Head> head_vector;
	head_vector m_heads;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API Device(IDXGIAdapter1 *pAdapter, const DeviceDesc &desc, const SwapChainDesc *swapChains, uint4 outputID = 0);
	/// Constructor.
	BE_GRAPHICS_DX11_API Device(ID3D11Device *pDevice);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~Device();

	/// Gets the number of device heads.
	BE_GRAPHICS_DX11_API uint4 GetHeadCount() const;
	/// Gets the swap chain of the device head identified by the given number.
	BE_GRAPHICS_DX11_API SwapChain* GetHeadSwapChain(uint4 headID) const;
	/// Gets a viewport corresponding to the requested device head's virtual desktop position.
	BE_GRAPHICS_DX11_API Viewport GetVirtualHeadViewport(uint4 headID) const;

	/// Presents the rendered image.
	BE_GRAPHICS_DX11_API void Present(bool bVSync = false);

	/// Gets the device feature level.
	BE_GRAPHICS_DX11_API uint4 GetFeatureLevel() const;

	/// Gets the D3D device.
	LEAN_INLINE ID3D11Device*const& GetInterface() const { return m_pDevice.get(); }
	/// Gets the D3D device.
	LEAN_INLINE ID3D11Device*const& GetDevice() const { return m_pDevice.get(); }

	/// Gets the implementation identifier.
	LEAN_INLINE ImplementationID GetImplementationID() const { return DX11Implementation; }
};

template <> struct ToImplementationDX11<beGraphics::Device> { typedef Device Type; };

} // namespace

} // namespace

#endif