/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_ADAPTERS
#define BE_GRAPHICS_ADAPTERS

#include "beGraphics.h"
#include "beFormat.h"
#include <lean/tags/noncopyable.h>
#include <beCore/beExchangeContainers.h>
#include <lean/smart/resource_ptr.h>
#include "beDevice.h"

namespace beGraphics
{

namespace Exchange = beCore::Exchange;

/// Graphics adapter interface.
class Adapter : public lean::noncopyable, public beCore::Resource, public Implementation
{
public:
	virtual ~Adapter() { }

	/// Display mode list.
	typedef Exchange::vector_t<DisplayMode>::t display_mode_vector;
	
	/// Gets the number of outputs.
	virtual uint4 GetOutputCount() const = 0;
	/// Gets whether the given format is supported for the given output.
	virtual bool IsFormatSupported(lean::uint4 outputID, Format::T format) = 0;
	/// Gets the display modes available for the given output.
	virtual display_mode_vector GetDisplayModes(lean::uint4 outputID, Format::T format, bool ignoreRefresh = true) const = 0;

	/// Gets the adapter's name.
	virtual Exchange::utf8_string GetName() const = 0;
};

/// Graphics interface.
class Graphics : public lean::noncopyable, public beCore::Resource, public Implementation
{
public:
	virtual ~Graphics() { }

	/// Creates a device for the given adapter.
	virtual lean::resource_ptr<Device, true> CreateDevice(const DeviceDesc &desc, const SwapChainDesc *swapChains,
		uint4 adapterID = 0, uint4 outputID = 0) const = 0;

	/// Gets the number of adapters available.
	virtual uint4 GetAdapterCount() const = 0; 
	/// Gets the adapter identified by the given number.
	virtual lean::resource_ptr<Adapter, 1> GetAdapter(uint4 adapterID) const = 0; 
};

/// Creates a graphics object.
BE_GRAPHICS_API lean::resource_ptr<Graphics, true> GetGraphics();

} // namespace

#endif