#include "Precompiled.h"
#include "Wrap3D/Src/BufferConfig.h"

#include "D3D10Buffer_Vertex.h"

namespace Mod
{
	D3D10Buffer_Vertex::D3D10Buffer_Vertex( const BufConfigType& cfg, ID3D10Device* dev ) : 
	Parent( cfg, D3D10_BIND_VERTEX_BUFFER, dev )
	{
	}

	//------------------------------------------------------------------------

	D3D10Buffer_Vertex::~D3D10Buffer_Vertex()
	{
	}

	//------------------------------------------------------------------------

}

