#include "Precompiled.h"

#include "Wrap3D/Src/BufferConfig.h"

#include "D3D9Format.h"
#include "D3D9Usage.h"
#include "D3D9IndexBuffer.h"

namespace Mod
{
	D3D9IndexBuffer::D3D9IndexBuffer( const IndexBufferConfig& cfg, IDirect3DDevice9* dev ) :
	Parent( cfg ),
	mLocked( false )
	{		
		const D3D9Usage *usg = static_cast<const D3D9Usage*>( cfg.usage );

		IDirect3DIndexBuffer9* res;
		MD_D3DV( dev->CreateIndexBuffer( (UINT)cfg.byteSize, usg->GetConfig().bufferUsage, static_cast< const D3D9Format*>(cfg.fmt)->GetIBFormat(), usg->GetConfig().bufPool, &res, NULL ) );

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

	D3D9IndexBuffer::~D3D9IndexBuffer()
	{
		if( mLocked )
		{
			UnmapImpl();
		}
	}
	
	//------------------------------------------------------------------------

	IDirect3DIndexBuffer9*
	D3D9IndexBuffer::ibuf() const
	{
		return static_cast< IDirect3DIndexBuffer9* >( GetResourceInternal() );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void D3D9IndexBuffer::MapImpl( void **ptr, MapType type ) /*OVERRIDE*/
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

		ibuf()->Lock( 0, (UINT)cfg.byteSize, ptr, static_cast< const D3D9Usage*>(cfg.usage)->GetConfig().lockFlags | extraLockFlags );
		mLocked = true;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9IndexBuffer::UnmapImpl() /*OVERRIDE*/
	{
		MD_FERROR_ON_FALSE( mLocked );
		ibuf()->Unlock();
		mLocked = false;
	}
	
	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9IndexBuffer::BindAsIBImpl( IDirect3DDevice9* dev ) const
	{
		dev->SetIndices( ibuf() );
	}



}