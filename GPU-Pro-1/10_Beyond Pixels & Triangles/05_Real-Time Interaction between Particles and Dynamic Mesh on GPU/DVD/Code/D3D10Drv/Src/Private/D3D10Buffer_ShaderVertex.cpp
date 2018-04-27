#include "Precompiled.h"
#include "D3D10Buffer_ShaderVertex.h"

namespace Mod
{
	D3D10Buffer_ShaderVertex::D3D10Buffer_ShaderVertex( const BufConfigType& cfg, ID3D10Device* dev ) : 
	Parent( cfg, D3D10_BIND_SHADER_RESOURCE | D3D10_BIND_VERTEX_BUFFER, dev )
	{
	}

	//------------------------------------------------------------------------

	D3D10Buffer_ShaderVertex::~D3D10Buffer_ShaderVertex()
	{
	}
}