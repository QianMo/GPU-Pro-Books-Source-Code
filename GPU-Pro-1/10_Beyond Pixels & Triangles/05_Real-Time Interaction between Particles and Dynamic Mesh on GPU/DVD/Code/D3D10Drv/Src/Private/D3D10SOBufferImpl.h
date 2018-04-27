#ifndef D3D10DRV_D3D10SOBUFFERIMPL_H_INCLUDED
#define D3D10DRV_D3D10SOBUFFERIMPL_H_INCLUDED

#include "D3D10BufferImpl.h"

namespace Mod
{
	template<typename Config>
	class D3D10SOBufferImpl : public D3D10BufferImpl<Config>
	{
		// types
	public:
		typedef D3D10BufferImpl<Config>				Base;
		typedef D3D10SOBufferImpl<Config>			Parent; // we're this parent

		// construction/ destruction
	public:
		D3D10SOBufferImpl ( const BufConfigType& cfg, UINT32 bindFlags, ID3D10Device* dev );
		~D3D10SOBufferImpl ();

		// polymorphism
	private:
		virtual void BindToImpl( SOBindSlot& slot ) const OVERRIDE;

	};
}

#endif