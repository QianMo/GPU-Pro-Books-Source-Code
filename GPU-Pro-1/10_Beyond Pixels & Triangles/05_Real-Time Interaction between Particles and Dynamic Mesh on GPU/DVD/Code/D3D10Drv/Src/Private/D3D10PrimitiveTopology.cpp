#include "Precompiled.h"
#include "D3D10PrimitiveTopology.h"

namespace Mod
{
	D3D10PrimitiveTopology::D3D10PrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY value ) :
	mValue( value )
	{

	}

	//------------------------------------------------------------------------

	D3D10PrimitiveTopology::~D3D10PrimitiveTopology()
	{

	}

	//------------------------------------------------------------------------
	
	D3D10_PRIMITIVE_TOPOLOGY
	D3D10PrimitiveTopology::GetD3D10Value() const
	{
		return mValue;
	}

}