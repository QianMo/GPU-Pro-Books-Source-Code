#include "Precompiled.h"
#include "D3D9Viewport.h"

#include "Wrap3D/Src/ViewportConfig.h"

namespace Mod
{
	D3D9Viewport::D3D9Viewport( const ViewportConfig& cfg ) : 
	Parent ( cfg )
	{
		mViewport.X				= cfg.topLeftX;
		mViewport.Y				= cfg.topLeftY;
		mViewport.Width			= cfg.width;
		mViewport.Height		= cfg.height;
		mViewport.MinZ			= cfg.minDepth;
		mViewport.MaxZ			= cfg.maxDepth;
	}

	//------------------------------------------------------------------------

	D3D9Viewport::~D3D9Viewport()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D9Viewport::BindTo( IDirect3DDevice9* dev ) const
	{
		dev->SetViewport( &mViewport );
	}

	//------------------------------------------------------------------------

	void
	D3D9Viewport::SetBindToZero( IDirect3DDevice9* dev )
	{
		D3DVIEWPORT9 viewport;

		viewport.X			= 0;
		viewport.Y			= 0;
		viewport.Height		= 0;
		viewport.Width		= 0;
		viewport.MinZ		= 1;
		viewport.MaxZ		= 1;

		dev->SetViewport( &viewport );
	}

	//------------------------------------------------------------------------


}