#ifndef D3D10DRV_D3D10BUFFER_SOSHADERVERTEX_H_INCLUDED
#define D3D10DRV_D3D10BUFFER_SOSHADERVERTEX_H_INCLUDED

#include "D3D10VertexBufferImpl.h"
#include "D3D10SOBufferImpl.h"

namespace Mod
{

	struct D3D10Buffer_SOShaderVertexConfig
	{
		typedef class D3D10Buffer_SOShaderVertex	Child;
		typedef SOShaderVertexBufferConfig			BufConfigType;
	};

	class D3D10Buffer_SOShaderVertex : public D3D10VertexBufferImpl< D3D10SOBufferImpl< D3D10Buffer_SOShaderVertexConfig > >
	{
	public:
		D3D10Buffer_SOShaderVertex( const BufConfigType& cfg, ID3D10Device* dev );
		virtual ~D3D10Buffer_SOShaderVertex();

	};

}

#endif