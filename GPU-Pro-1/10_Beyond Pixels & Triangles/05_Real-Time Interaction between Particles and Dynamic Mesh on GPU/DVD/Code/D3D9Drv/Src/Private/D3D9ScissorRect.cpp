#include "Precompiled.h"

#include "Wrap3D/Src/ScissorRectConfig.h"

#include "D3D9ScissorRect.h"

namespace Mod
{
	D3D9ScissorRect::D3D9ScissorRect( const ScissorRectConfig& cfg ) :
	Parent ( cfg )
	{

	}

	//------------------------------------------------------------------------

	D3D9ScissorRect::~D3D9ScissorRect()
	{

	}

	//------------------------------------------------------------------------
	
	void
	D3D9ScissorRect::BindTo( IDirect3DDevice9* dev ) const
	{
		const ScissorRectConfig& cfg = GetConfig();

		RECT rect;

		rect.left	= cfg.left;
		rect.top	= cfg.top;
		rect.right	= cfg.right;
		rect.bottom	= cfg.bottom;

		dev->SetScissorRect( &rect );
	}

	//------------------------------------------------------------------------
	/*static*/

	void
	D3D9ScissorRect::SetBindToZero( IDirect3DDevice9* dev )
	{
		RECT rect;

		rect.left	= 0;
		rect.top	= 0;
		rect.right	= 0;
		rect.bottom	= 0;

		dev->SetScissorRect( &rect );
	}

	//------------------------------------------------------------------------

}

