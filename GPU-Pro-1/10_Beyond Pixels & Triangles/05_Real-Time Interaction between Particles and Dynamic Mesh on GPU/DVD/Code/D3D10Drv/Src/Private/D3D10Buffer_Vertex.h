#ifndef D3D10DRV_D3D10BUFFER_VERTEX_H_INCLUDED
#define D3D10DRV_D3D10BUFFER_VERTEX_H_INCLUDED

#include "D3D10VertexBufferImpl.h"

namespace Mod
{

	struct D3D10Buffer_VertexConfig
	{
		typedef class D3D10Buffer_Vertex	Child;
		typedef VertexBufferConfig			BufConfigType;
	};

	class D3D10Buffer_Vertex : public D3D10VertexBufferImpl< D3D10BufferImpl<D3D10Buffer_VertexConfig> >
	{
	public:
		D3D10Buffer_Vertex( const BufConfigType& cfg, ID3D10Device* dev );
		virtual ~D3D10Buffer_Vertex();


	};
}

#endif