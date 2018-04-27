#include "Precompiled.h"

#include "Wrap3D/Src/ScissorRectConfig.h"

#include "D3D10ScissorRect.h"

namespace Mod
{
	D3D10ScissorRect::D3D10ScissorRect( const ScissorRectConfig& cfg ) :
	Parent ( cfg )
	{

	}

	//------------------------------------------------------------------------

	D3D10ScissorRect::~D3D10ScissorRect()
	{

	}

	//------------------------------------------------------------------------
	
	void
	D3D10ScissorRect::BindTo( BindType& slot ) const
	{
		const ScissorRectConfig& cfg = GetConfig();

		slot.left	= cfg.left;
		slot.top	= cfg.top;
		slot.right	= cfg.right;
		slot.bottom	= cfg.bottom;
	}

	//------------------------------------------------------------------------

	void
	D3D10ScissorRect::SetBindToZero( BindType& slot )
	{
		slot.left	= 0;
		slot.top	= 0;
		slot.right	= 0;
		slot.bottom	= 0;
	}

	//------------------------------------------------------------------------

	bool operator ! ( const D3D10_RECT& rect )
	{
		return !rect.left && !rect.right && !rect.bottom && !rect.top;
	}
}

