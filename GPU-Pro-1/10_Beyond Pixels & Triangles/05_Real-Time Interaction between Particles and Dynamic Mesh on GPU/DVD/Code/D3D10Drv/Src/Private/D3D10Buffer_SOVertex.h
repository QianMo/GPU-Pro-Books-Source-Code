#ifndef D3D10DRV_D3D10BUFFER_SOVERTEX_H_INCLUDED
#define D3D10DRV_D3D10BUFFER_SOVERTEX_H_INCLUDED

#include "D3D10VertexBufferImpl.h"
#include "D3D10SOBufferImpl.h"

namespace Mod
{

	struct D3D10Buffer_SOVertexConfig
	{
		typedef class D3D10Buffer_SOVertex		Child;
		typedef SOVertexBufferConfig			BufConfigType;
	};

	class D3D10Buffer_SOVertex : public D3D10VertexBufferImpl< D3D10SOBufferImpl< D3D10Buffer_SOVertexConfig > >
	{
	public:
		D3D10Buffer_SOVertex( const BufConfigType& cfg, ID3D10Device* dev );
		virtual ~D3D10Buffer_SOVertex();

	};

}

#endif