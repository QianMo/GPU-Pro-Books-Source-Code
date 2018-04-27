#include "Precompiled.h"
#include "D3D10Buffer_SOShaderVertex.h"

namespace Mod
{
	D3D10Buffer_SOShaderVertex::D3D10Buffer_SOShaderVertex( const BufConfigType& cfg, ID3D10Device* dev ) : 
	Parent( cfg, D3D10_BIND_VERTEX_BUFFER | D3D10_BIND_STREAM_OUTPUT | D3D10_BIND_SHADER_RESOURCE, dev )
	{
	}

	//------------------------------------------------------------------------

	D3D10Buffer_SOShaderVertex::~D3D10Buffer_SOShaderVertex()
	{
	}
}