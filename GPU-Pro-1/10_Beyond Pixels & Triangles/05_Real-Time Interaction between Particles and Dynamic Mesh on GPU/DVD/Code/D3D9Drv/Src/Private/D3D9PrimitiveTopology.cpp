#include "Precompiled.h"
#include "D3D9PrimitiveTopology.h"

namespace Mod
{
	D3D9PrimitiveTopology::D3D9PrimitiveTopology( D3DPRIMITIVETYPE value ) :
	mValue( value )
	{

	}

	//------------------------------------------------------------------------

	D3D9PrimitiveTopology::~D3D9PrimitiveTopology()
	{

	}

	//------------------------------------------------------------------------

	D3DPRIMITIVETYPE
	D3D9PrimitiveTopology::GetD3D9Value() const
	{
		return mValue;
	}

}