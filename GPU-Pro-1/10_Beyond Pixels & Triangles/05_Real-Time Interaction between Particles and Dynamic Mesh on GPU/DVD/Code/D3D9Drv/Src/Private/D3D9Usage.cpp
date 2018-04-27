#include "Precompiled.h"

#include "D3D9Usage.h"

namespace Mod
{
	D3D9Usage::D3D9Usage( const D3D9UsageConfig& cfg ) :
	mConfig( cfg )
	{

	}

	//------------------------------------------------------------------------

	/*virtual*/
	D3D9Usage::~D3D9Usage()
	{

	}

	//------------------------------------------------------------------------

	const D3D9UsageConfig&
	D3D9Usage::GetConfig() const
	{
		return mConfig;
	}



}