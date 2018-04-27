#ifndef PROVIDERS_MODELVERTEXBUFFERS_H_INCLUDED
#define PROVIDERS_MODELVERTEXBUFFERS_H_INCLUDED

#include "Forw.h"

namespace Mod
{
	struct ModelVertexBuffers
	{
		typedef TypesI< BufferPtr, BT::BUFFERTYPE_COUNT > :: StaticArray VertexBuffers;

		VertexBuffers	vertexBuffers;
		bool			isSplit;

		explicit ModelVertexBuffers( bool isSplit );
		const BufferPtr& operator[] ( UINT32 idx ) const;
	};

	//------------------------------------------------------------------------

	inline
	ModelVertexBuffers::ModelVertexBuffers( bool a_isSplit ) :
	isSplit( a_isSplit )
	{

	}

	//------------------------------------------------------------------------

	inline
	const
	BufferPtr&
	ModelVertexBuffers::operator[] ( UINT32 idx ) const
	{
		return vertexBuffers[ idx ];
	}
}

#endif
