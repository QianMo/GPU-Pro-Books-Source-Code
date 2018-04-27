#include "Precompiled.h"

#include "Wrap3D/Src/BufferConfig.h"

#include "D3D9Usage.h"
#include "D3D9VertexBuffer.h"

namespace Mod
{
	D3D9VertexBuffer::D3D9VertexBuffer( const VertexBufferConfig& cfg, IDirect3DDevice9* dev ) :
	Parent( cfg ),
	mLocked( false )
	{		
		const D3D9Usage *usg = static_cast<const D3D9Usage*>( cfg.usage );

		IDirect3DVertexBuffer9* res;
		MD_D3DV( dev->CreateVertexBuffer( (UINT)cfg.byteSize, usg->GetConfig().bufferUsage, 0, usg->GetConfig().bufPool, &res, NULL ) );

		if( cfg.data.GetSize() )
		{
			void* ptr;
			MD_D3DV( res->Lock( 0, (UINT)cfg.data.GetSize(), &ptr, 0 ) );
			memcpy( ptr, cfg.data.GetRawPtr(), (UINT)cfg.data.GetSize() );
			MD_D3DV( res->Unlock() );
		}

		SetResource( res );
	}

	//------------------------------------------------------------------------

	D3D9VertexBuffer::~D3D9VertexBuffer()
	{
		if( mLocked )
		{
			UnmapImpl();
		}
	}
	
	//------------------------------------------------------------------------

	IDirect3DVertexBuffer9*
	D3D9VertexBuffer::vbuf() const
	{
		return static_cast< IDirect3DVertexBuffer9* >( GetResourceInternal() );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void D3D9VertexBuffer::MapImpl( void **ptr, MapType type ) /*OVERRIDE*/
	{
		MD_FERROR_ON_TRUE( mLocked );
		const ConfigType& cfg = GetConfig();

		UINT extraLockFlags( 0 );

		if( type == MAP_DISCARD )
		{
			extraLockFlags |= D3DLOCK_DISCARD;
		}
		else
		if( type == MAP_NO_OVERWRITE )
		{
			extraLockFlags |= D3DLOCK_NOOVERWRITE;
		}


		MD_D3DV( vbuf()->Lock( 0, 0, ptr, static_cast< const D3D9Usage*>(cfg.usage)->GetConfig().lockFlags | extraLockFlags ) );
		mLocked = true;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9VertexBuffer::UnmapImpl() /*OVERRIDE*/
	{
		MD_FERROR_ON_FALSE( mLocked );
		MD_D3DV( vbuf()->Unlock() );
		mLocked = false;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9VertexBuffer::BindAsVBImpl( IDirect3DDevice9* dev, UINT32 slot ) const /*OVERRIDE*/
	{
		MD_D3DV( dev->SetStreamSource( slot, vbuf(), 0, static_cast<const VertexBufferConfig&>( GetConfig() ).stride ) );
	}


}