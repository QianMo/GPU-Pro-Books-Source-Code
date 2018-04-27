#ifndef D3D10DRV_D3D10BUFFER_SHADERVERTEX_H_INCLUDED
#define D3D10DRV_D3D10BUFFER_SHADERVERTEX_H_INCLUDED

#include "D3D10BufferImpl.h"

namespace Mod
{

	struct D3D10Buffer_ShaderVertexConfig
	{
		typedef class D3D10Buffer_ShaderVertex		Child;
		typedef ShaderVertexBufferConfig			BufConfigType;
	};

	class D3D10Buffer_ShaderVertex : public D3D10BufferImpl< D3D10Buffer_ShaderVertexConfig >
	{
	public:
		D3D10Buffer_ShaderVertex( const BufConfigType& cfg, ID3D10Device* dev );
		virtual ~D3D10Buffer_ShaderVertex();

	};

}

#endif