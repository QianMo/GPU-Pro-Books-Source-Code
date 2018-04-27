#include "Precompiled.h"
#include "D3D10Buffer_SOShader.h"

namespace Mod
{
	D3D10Buffer_SOShader::D3D10Buffer_SOShader( const BufConfigType& cfg, ID3D10Device* dev ) : 
	Parent( cfg, D3D10_BIND_SHADER_RESOURCE | D3D10_BIND_STREAM_OUTPUT, dev )
	{
	}

	//------------------------------------------------------------------------

	D3D10Buffer_SOShader::~D3D10Buffer_SOShader()
	{
	}
}