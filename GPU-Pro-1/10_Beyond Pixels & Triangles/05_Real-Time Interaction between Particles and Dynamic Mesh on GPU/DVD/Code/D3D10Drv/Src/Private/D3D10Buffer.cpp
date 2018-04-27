#include "Precompiled.h"
#include "Precompiled.h"
#include "D3D10Buffer.h"
#include "D3D10Usage.h"
#include "D3D10Exception.h"

#include "Wrap3D\Src\BufferConfig.h"

namespace Mod
{
	D3D10Buffer::D3D10Buffer( const BufferConfig& cfg, UINT32 bindFlags, ID3D10Device* dev ) : 
	Base( cfg ),
	mBindFlags( bindFlags )
	{
		D3D10_BUFFER_DESC desc;

		const D3D10Usage* usg	= static_cast<const D3D10Usage*>(cfg.usage);

		desc.ByteWidth			= static_cast<UINT>( cfg.data.GetSize() ? cfg.data.GetSize(): cfg.byteSize );
		desc.Usage				= usg->GetValue();
		desc.BindFlags			= bindFlags;
		desc.CPUAccessFlags		= usg->GetDefaultAccessFlags();
		desc.MiscFlags			= 0;

		{
			ID3D10Buffer* buf;

			D3D10_SUBRESOURCE_DATA sdata = {};

			sdata.pSysMem = cfg.data.GetRawPtr();

			D3D10_THROW_IF( dev->CreateBuffer( &desc, sdata.pSysMem ? &sdata : NULL, &buf ) );
			mResource.set( buf );
		}
	}

	//------------------------------------------------------------------------

	D3D10Buffer::~D3D10Buffer()
	{
	}

	//------------------------------------------------------------------------

	const
	D3D10Buffer::ResourcePtr&
	D3D10Buffer::GetResource() const
	{
		return mResource;
	}

	//------------------------------------------------------------------------

	UINT32
	D3D10Buffer::GetBindFlags() const
	{
		return mBindFlags;
	}

	//------------------------------------------------------------------------

	void
	D3D10Buffer::BindTo( IABindSlot& slot ) const
	{
		*slot.buffer = &*mResource;
		BindToImpl( slot );
	}

	//------------------------------------------------------------------------

	void
	D3D10Buffer::SetBindToZero( IABindSlot& slot )
	{
		*slot.buffer = NULL;
		*slot.offset = 0;
		*slot.stride = 0;
	}

	//------------------------------------------------------------------------

	void
	D3D10Buffer::BindTo( SOBindSlot& slot ) const
	{
		*slot.buffer	= &*mResource;
		slot.set		= true;
		BindToImpl( slot );
	}

	//------------------------------------------------------------------------

	void
	D3D10Buffer::SetBindToZero( SOBindSlot& slot )
	{
		*slot.buffer	= NULL;
		*slot.offset	= 0;
		slot.set		= true;
	}

	//------------------------------------------------------------------------

	void
	D3D10Buffer::SetResource( ResourcePtr::PtrType res )
	{
		mResource.set( res );
	}

	//------------------------------------------------------------------------

	D3D10Buffer::ResourcePtr::PtrType
	D3D10Buffer::GetResourceInternal() const
	{
		return &*mResource;
	}

	//------------------------------------------------------------------------

	void
	D3D10Buffer::BindTo( ID3D10EffectConstantBuffer* slot ) const
	{
		BindToImpl( slot );
	}

	//------------------------------------------------------------------------

	void
	D3D10Buffer::SetBindToZero( ID3D10EffectConstantBuffer* slot )
	{
		slot->SetConstantBuffer( NULL );
	}


	//------------------------------------------------------------------------

	void
	D3D10Buffer::BindToImpl( IABindSlot& slot ) const
	{		
		slot;
		MD_FERROR( L"D3D10Buffer::BindToImpl: attempt to bind non-vertex buffer to IA stage!" );
	}

	//------------------------------------------------------------------------

	void
	D3D10Buffer::BindToImpl( SOBindSlot& slot ) const
	{
		slot;
		MD_FERROR( L"D3D10Buffer::BindToImpl: attempt to bind non-so buffer to SO stage!" );
	}

	//------------------------------------------------------------------------

	void
	D3D10Buffer::BindToImpl( ID3D10EffectConstantBuffer* slot ) const
	{
		slot;
		MD_FERROR( L"D3D10Buffer::BindToImpl: attempt to bind non-constant buffer to effect!" );
	}

	//------------------------------------------------------------------------
	void
	D3D10Buffer::MapImpl( void ** ptr, MapType type )
	{
		D3D10_MAP mapType( D3D10_MAP_READ );

		if( UINT32 accessFlags = static_cast<const D3D10Usage*>( GetConfig().usage )->GetDefaultAccessFlags() )
		{
			if( accessFlags & D3D10_CPU_ACCESS_READ )
			{
				if( accessFlags & D3D10_CPU_ACCESS_WRITE )
					mapType = D3D10_MAP_READ_WRITE;
				else
					mapType = D3D10_MAP_READ;
			}
			else
			if( accessFlags & D3D10_CPU_ACCESS_WRITE )
			{
				switch( type )
				{
				case MAP_DISCARD:
					mapType = D3D10_MAP_WRITE_DISCARD;
					break;
				case MAP_NO_OVERWRITE:
					mapType = D3D10_MAP_WRITE_NO_OVERWRITE;
					break;
				}
			}
		}
		else
			MD_FERROR( L"D3D10Buffer::MapImpl: cannot map with default access flags!" );

		mResource->Map( mapType, 0, ptr );		
	}

	//------------------------------------------------------------------------
	void
	D3D10Buffer::UnmapImpl()
	{
		mResource->Unmap();
	}

	//------------------------------------------------------------------------

	bool
	D3D10Buffer::IABindSlot::operator ! () const
	{
		return *buffer ? false : true;
	}

	//------------------------------------------------------------------------

	D3D10Buffer::SOBindSlot::SOBindSlot() :
	set( false )
	{

	}

	//------------------------------------------------------------------------

	bool
	D3D10Buffer::SOBindSlot::operator ! () const
	{
		return !set;
	}


}