#include "Precompiled.h"
#include "D3D10Buffer_SOVertex.h"

namespace Mod
{
	D3D10Buffer_SOVertex::D3D10Buffer_SOVertex( const BufConfigType& cfg, ID3D10Device* dev ) : 
	Parent( cfg, D3D10_BIND_VERTEX_BUFFER | D3D10_BIND_STREAM_OUTPUT, dev )
	{
	}

	//------------------------------------------------------------------------

	D3D10Buffer_SOVertex::~D3D10Buffer_SOVertex()
	{
	}
}