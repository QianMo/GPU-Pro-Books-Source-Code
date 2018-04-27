#ifndef D3D9DRV_D3D9VERTEXBUFFER_H_INCLUDED
#define D3D9DRV_D3D9VERTEXBUFFER_H_INCLUDED

#include "Precompiled.h"

#include "D3D9Buffer.h"

namespace Mod
{
	class D3D9VertexBuffer : public D3D9Buffer
	{
		// construction/ destruction
	public:
		D3D9VertexBuffer( const VertexBufferConfig& cfg, IDirect3DDevice9* dev );
		~D3D9VertexBuffer();


		// helpers
	private:
		IDirect3DVertexBuffer9* vbuf() const;

		// polymorphism
	private:
		virtual void MapImpl( void **ptr, MapType type ) OVERRIDE;
		virtual void UnmapImpl() OVERRIDE;

		virtual void BindAsVBImpl( IDirect3DDevice9* dev, UINT32 slot ) const OVERRIDE;

		// data
	private:
		bool mLocked;	

	};
}

#endif