#include "Precompiled.h"
#include "D3D10Usage.h"

namespace Mod
{

	D3D10Usage::D3D10Usage( D3D10_USAGE usage, INT32 defCPUAccessFlags ):
	mD3D10Usage( usage ),
	mDefaultCPUAccessFlags( defCPUAccessFlags )
	{

	}

	//------------------------------------------------------------------------

	D3D10Usage::~D3D10Usage()
	{

	}

	//------------------------------------------------------------------------

	D3D10_USAGE
	D3D10Usage::GetValue() const
	{
		return mD3D10Usage;
	}

	//------------------------------------------------------------------------

	INT32
	D3D10Usage::GetDefaultAccessFlags() const
	{
		return mDefaultCPUAccessFlags;
	}

}