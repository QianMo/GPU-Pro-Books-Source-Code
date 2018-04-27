#ifndef D3D9DRV_D3D9INDEXBUFFER_H_INCLUDED
#define D3D9DRV_D3D9INDEXBUFFER_H_INCLUDED

#include "Precompiled.h"

#include "D3D9Buffer.h"

namespace Mod
{
	class D3D9IndexBuffer : public D3D9Buffer
	{
		// construction/ destruction
	public:
		D3D9IndexBuffer( const IndexBufferConfig& cfg, IDirect3DDevice9* dev );
		~D3D9IndexBuffer();


		// helpers
	private:
		IDirect3DIndexBuffer9* ibuf() const;

		// polymorphism
	private:
		virtual void MapImpl( void **ptr, MapType type ) OVERRIDE;
		virtual void UnmapImpl() OVERRIDE;

		virtual void BindAsIBImpl( IDirect3DDevice9* dev ) const OVERRIDE;

		// data
	private:
		bool mLocked;	

	};
}

#endif