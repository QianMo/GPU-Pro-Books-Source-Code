#include "Precompiled.h"
#include "D3D10Viewport.h"

#include "Wrap3D/Src/ViewportConfig.h"

namespace Mod
{
	D3D10Viewport::D3D10Viewport( const ViewportConfig& cfg ) : 
	Parent ( cfg )
	{
		mViewport.TopLeftX		= cfg.topLeftX;
		mViewport.TopLeftY		= cfg.topLeftY;
		mViewport.Width			= cfg.width;
		mViewport.Height		= cfg.height;
		mViewport.MinDepth		= cfg.minDepth;
		mViewport.MaxDepth		= cfg.maxDepth;
	}

	//------------------------------------------------------------------------

	D3D10Viewport::~D3D10Viewport()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D10Viewport::BindTo( BindType& slot ) const
	{
		slot = mViewport;
	}

	//------------------------------------------------------------------------

	void
	D3D10Viewport::SetBindToZero( BindType& slot )
	{
		slot.TopLeftX		= 0;
		slot.TopLeftY		= 0;
		slot.Height			= 0;
		slot.Width			= 0;
		slot.MinDepth		= 1;
		slot.MaxDepth		= 1;
	}

	//------------------------------------------------------------------------

	bool operator ! ( const D3D10_VIEWPORT& vp )
	{
		return !vp.Width;
	}

}