#include "Precompiled.h"
#include "D3D10Buffer_Shader.h"


namespace Mod
{

	D3D10Buffer_Shader::D3D10Buffer_Shader( const BufConfigType& cfg, ID3D10Device* dev ) :
	Parent( cfg, D3D10_BIND_SHADER_RESOURCE, dev )
	{
	}

	//------------------------------------------------------------------------

	D3D10Buffer_Shader::~D3D10Buffer_Shader()
	{
	}

}
