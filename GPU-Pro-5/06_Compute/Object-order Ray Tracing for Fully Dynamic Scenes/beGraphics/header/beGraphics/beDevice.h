/*******************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*******************************************************/

#pragma once
#ifndef BE_GRAPHICS_DEVICE
#define BE_GRAPHICS_DEVICE

#include "beGraphics.h"
#include "beFormat.h"
#include <lean/tags/noncopyable.h>
#include <beCore/beShared.h>
#include <lean/smart/resource_ptr.h>
#include <lean/smart/com_ptr.h>

namespace beGraphics
{

/// Device type enumeration.
struct DeviceType
{
	/// Enumeration.
	enum T
	{
		Hardware,	///< Hardware device
		Software,	///< Software device
		End
	};
	LEAN_MAKE_ENUM_STRUCT(DeviceType)
};

/// Device description.
struct DeviceDesc
{
	void *Window;			///< Platform-specific window handle.	
	bool Windowed;			///< Specifies whether to create windowed or full-screen device.
	uint4 FeatureLevel;		///< Minimum shader version.
	DeviceType::T Type;		///< Device type.
	bool MultiHead;			///< Specifies whether to acquire all outputs connected to the adapter.
	bool Debug;

	/// Constructor.
	explicit DeviceDesc(
		void *window = nullptr,		
		bool windowed = true,
		uint4 featureLevel = 0,
		DeviceType::T type = DeviceType::Hardware,
		bool multiHead = false,
		bool debug = false)
			: Window(window),
			Windowed(windowed),
			FeatureLevel(featureLevel),
			Type(type),
			MultiHead(multiHead),
			Debug(debug) { }
};

/// Swap chain description.
struct SwapChainDesc
{
	DisplayMode Display;	///< Display mode.
	uint4 BufferCount;		///< Number of buffers.
	void *Window;			///< Platform-specific window handle.
	bool Windowed;			///< Specifies whether to create windowed or full-screen device.
	SampleDesc Samples;		///< Multisampling options.

	/// Constructor.
	explicit SwapChainDesc(const DisplayMode &display = DisplayMode(),
		uint4 bufferCount = 1,
		void *window = nullptr,
		bool windowed = true,
		SampleDesc samples = SampleDesc())
			: Display(display),
			BufferCount(bufferCount),
			Window(window),
			Windowed(windowed),
			Samples(samples) { }
};

/// Constructs a swap chain description from the given descriptions.
inline SwapChainDesc ToSwapChainDesc(const DeviceDesc &deviceDesc, const SwapChainDesc &swapChainDesc)
{
	return SwapChainDesc(
		swapChainDesc.Display,
		swapChainDesc.BufferCount,
		(swapChainDesc.Window != NULL) ? swapChainDesc.Window : deviceDesc.Window,
		deviceDesc.Windowed,
		swapChainDesc.Samples );
}

/// Swap chain interfaec
class SwapChain : public beCore::Resource, public Implementation
{
protected:
	LEAN_INLINE SwapChain& operator =(const SwapChain&) { return *this; }

public:
	virtual ~SwapChain() { }

	/// Presents the rendered image.
	virtual void Present(bool bVSync = false) = 0;

	/// Resizes the swap chain buffers.
	virtual void Resize(uint4 width, uint4 height) = 0;

	/// Gets the swap chain description.
	virtual SwapChainDesc GetDesc() const = 0;
};

class Device;

/// Creates a swap chain for the given device.
BE_GRAPHICS_API lean::resource_ptr<SwapChain, true> CreateSwapChain(const Device &device, const SwapChainDesc &desc);

/// Viewport.
struct Viewport
{
	float X;		///< Position.
	float Y;		///< Position.
	float Width;	///< Width.
	float Height;	///< Height.
	float MinDepth;	///< Minimum depth.
	float MaxDepth;	///< Maximum depth.

	/// Constructor.
	explicit Viewport(float x = 0.0f,
		float y = 0.0f,
		float width = 0.0f,
		float height = 0.0f,
		float minDepth = 0.0f,
		float maxDepth = 1.0f)
			: X(x),
			Y(y),
			Width(width),
			Height(height),
			MinDepth(minDepth),
			MaxDepth(maxDepth) { }
};

/// Device interface.
class Device : public beCore::Resource, public Implementation
{
protected:
	LEAN_INLINE Device& operator =(const Device&) { return *this; }

public:
	virtual ~Device() throw() { }

	/// Gets the number of device heads.
	virtual uint4 GetHeadCount() const = 0;
	/// Gets the swap chain of the device head identified by the given number.
	virtual SwapChain* GetHeadSwapChain(uint4 headID) const = 0;
	/// Gets a viewport corresponding to the requested device head's virtual desktop position.
	virtual Viewport GetVirtualHeadViewport(uint4 headID) const = 0;

	/// Presents the rendered image.
	virtual void Present(bool bVSync = false) = 0;

	/// Gets the device feature level.
	virtual uint4 GetFeatureLevel() const = 0;
};

} // namespace

#endif