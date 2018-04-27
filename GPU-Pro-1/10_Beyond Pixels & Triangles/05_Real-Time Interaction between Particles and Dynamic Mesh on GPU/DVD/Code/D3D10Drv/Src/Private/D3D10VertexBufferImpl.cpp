#include "Precompiled.h"

#include "Wrap3D\Src\BufferConfig.h"

#include "D3D10SOBufferImpl.h"
#include "D3D10VertexBufferImpl.h"

#include "D3D10Buffer_Vertex.h"
#include "D3D10Buffer_SOVertex.h"
#include "D3D10Buffer_SOShaderVertex.h"

namespace Mod
{
	template <typename BaseClass>
	D3D10VertexBufferImpl<BaseClass>::D3D10VertexBufferImpl( const typename Base::BufConfigType& cfg, UINT32 bindFlags, ID3D10Device* dev ) :
	Base( cfg, bindFlags, dev )
	{

	}

	//------------------------------------------------------------------------

	template <typename BaseClass>
	D3D10VertexBufferImpl<BaseClass>::~D3D10VertexBufferImpl ()
	{

	}

	//------------------------------------------------------------------------

	template <typename BaseClass>
	void
	D3D10VertexBufferImpl<BaseClass>::BindToImpl( typename Base::IABindSlot& slot ) const
	{

		MD_CHECK_TYPE(const BufConfigType,&GetConfig());

		const BufConfigType& cfg = static_cast<const BufConfigType&>(GetConfig());
		*slot.stride = cfg.stride;
		*slot.offset = 0;
	}

	//------------------------------------------------------------------------

	template class D3D10VertexBufferImpl< D3D10BufferImpl	< D3D10Buffer_Vertex::Config			> >;
	template class D3D10VertexBufferImpl< D3D10SOBufferImpl	< D3D10Buffer_SOVertex::Config			> >;
	template class D3D10VertexBufferImpl< D3D10SOBufferImpl	< D3D10Buffer_SOShaderVertex::Config	> >;

}