#ifndef D3D10DRV_D3D10VERTEXBUFFERIMPL_H_INCLUDED
#define D3D10DRV_D3D10VERTEXBUFFERIMPL_H_INCLUDED

#include "D3D10BufferImpl.h"

namespace Mod
{

	template<typename BaseClass>
	class D3D10VertexBufferImpl : public BaseClass
	{
		// types
	public:
		typedef BaseClass							Base;
		typedef D3D10VertexBufferImpl<BaseClass>	Parent; // we're this parent

		// construction/ destruction
	public:
		D3D10VertexBufferImpl( const typename Base::BufConfigType& cfg, UINT32 bindFlags, ID3D10Device* dev );
		~D3D10VertexBufferImpl ();

		// polymorphism
	private:
		virtual void BindToImpl( typename Base::IABindSlot& slot ) const OVERRIDE;

	};

}

#endif