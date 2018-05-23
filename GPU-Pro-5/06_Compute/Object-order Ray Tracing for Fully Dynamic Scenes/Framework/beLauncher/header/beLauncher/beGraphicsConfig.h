/*****************************************************/
/* breeze Framework Launch Lib  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_LAUNCHER_GRAPHICSCONFIG
#define BE_LAUNCHER_GRAPHICSCONFIG

#include "beLauncher.h"
#include <beGraphics/beAdapters.h>
#include <beGraphics/beDevice.h>
#include <Windows.h>

#include <lean/smart/resource_ptr.h>

namespace beLauncher
{

/// Graphics configuration
struct GraphicsConfig
{
	beGraphics::DeviceDesc DeviceDesc;					///< Device description.
	beGraphics::SwapChainDesc SwapChain;				///< Swap chain description.
	bool VSync;											///< V-Sync.
	lean::uint4 AdapterID;								///< Adapter ID.
	lean::uint4 OutputID;								///< Output ID.
	lean::uint4 OutputCount;							///< Number of outputs connected to the current adapter.
	lean::resource_ptr<beGraphics::Graphics> Graphics;	///< Graphics object.

	/// Constructor.
	GraphicsConfig(const beGraphics::DeviceDesc &deviceDesc = beGraphics::DeviceDesc(),
		const beGraphics::SwapChainDesc &swapChain = beGraphics::SwapChainDesc(beGraphics::DisplayMode()),
		bool vSync = false,
		lean::uint4 adapterID = 0,
		lean::uint4 outputID = 0,
		beGraphics::Graphics *graphics = nullptr,
		lean::uint4 outputCount = 1)
			: DeviceDesc(deviceDesc),
			SwapChain(swapChain),
			VSync(vSync),
			AdapterID(adapterID),
			OutputID(outputID),
			Graphics(graphics),
			OutputCount(outputCount) { }
};

/// Opens a graphics configuration dialog.
BE_LAUNCHER_API bool OpenGraphicsConfiguration(GraphicsConfig &config, HINSTANCE hInstance, HICON hIcon = NULL, HWND hParent = NULL);

}

#endif