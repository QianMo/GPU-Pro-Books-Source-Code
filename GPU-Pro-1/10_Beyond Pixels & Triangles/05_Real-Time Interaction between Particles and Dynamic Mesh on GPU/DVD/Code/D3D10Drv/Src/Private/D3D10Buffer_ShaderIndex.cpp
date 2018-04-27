#include "Precompiled.h"
#include "D3D10Buffer_ShaderIndex.h"

namespace Mod
{
	D3D10Buffer_ShaderIndex::D3D10Buffer_ShaderIndex( const BufConfigType& cfg, ID3D10Device* dev ) : 
	Parent( cfg, D3D10_BIND_SHADER_RESOURCE | D3D10_BIND_INDEX_BUFFER, dev )
	{
	}

	//------------------------------------------------------------------------

	D3D10Buffer_ShaderIndex::~D3D10Buffer_ShaderIndex()
	{
	}
}