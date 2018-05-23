/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_ADAPTERS_DX11
#define BE_GRAPHICS_ADAPTERS_DX11

#include "beGraphics.h"
#include "../beAdapters.h"
#include <lean/smart/com_ptr.h>
#include <DXGI.h>

namespace beGraphics
{

namespace DX11
{

/// Graphics adapter implementation.
class Adapter : public beGraphics::Adapter
{
private:
	lean::com_ptr<IDXGIAdapter1> m_pAdapter;

	typedef std::vector< lean::com_ptr<IDXGIOutput> > output_vector;
	output_vector m_outputs;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API Adapter(IDXGIAdapter1 *pAdapter);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~Adapter();
	
	/// Gets the number of outputs.
	BE_GRAPHICS_DX11_API uint4 GetOutputCount() const;
	/// Gets whether the given format is supported for the given output.
	BE_GRAPHICS_DX11_API bool IsFormatSupported(lean::uint4 outputID, Format::T format);
	// Gets the display modes available for the given output.
	BE_GRAPHICS_DX11_API display_mode_vector GetDisplayModes(lean::uint4 outputID, Format::T format, bool ignoreRefresh = true) const;

	/// Gets the adapter's name.
	BE_GRAPHICS_DX11_API Exchange::utf8_string GetName() const;

	/// Gets the DXGI adapter.
	LEAN_INLINE IDXGIAdapter1* GetAdapter() const { return m_pAdapter; }
	/// Gets the DXGI output identified by the given number.
	LEAN_INLINE IDXGIOutput* GetOutput(lean::uint4 outputID) const { return (outputID < m_outputs.size()) ? m_outputs[outputID] : nullptr; }

	/// Gets the implementation identifier.
	LEAN_INLINE ImplementationID GetImplementationID() const { return DX11Implementation; };
};

template <> struct ToImplementationDX11<beGraphics::Adapter> { typedef Adapter Type; };

/// Graphics implementation.
class Graphics : public beGraphics::Graphics
{
private:
	lean::com_ptr<IDXGIFactory1> m_pFactory;

	typedef std::vector< lean::com_ptr<IDXGIAdapter1> > adapter_vector;
	adapter_vector m_adapters;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API Graphics();
	/// Constructor.
	BE_GRAPHICS_DX11_API Graphics(IDXGIFactory1 *pFactory);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~Graphics();

	/// Creates a device for the given adapter.
	BE_GRAPHICS_DX11_API lean::resource_ptr<beGraphics::Device, true> CreateDevice(const DeviceDesc &desc, const SwapChainDesc *swapChains,
		uint4 adapterID = 0, uint4 outputID = 0) const;

	/// Gets the number of adapters available.
	BE_GRAPHICS_DX11_API uint4 GetAdapterCount() const;
	/// Gets the adapter identified by the given number.
	BE_GRAPHICS_DX11_API lean::resource_ptr<beGraphics::Adapter, true> GetAdapter(uint4 adapterID) const;

	/// Gets the DXGI factory.
	LEAN_INLINE IDXGIFactory1* GetFactory() const { return m_pFactory; }

	/// Gets the implementation identifier.
	LEAN_INLINE ImplementationID GetImplementationID() const { return DX11Implementation; };
};

template <> struct ToImplementationDX11<beGraphics::Graphics> { typedef Graphics Type; };

} // namespace

} // namespace

#endif