#include "Precompiled.h"

#include "D3D9Buffer.h"

namespace Mod
{
	D3D9Buffer::D3D9Buffer( const BufferConfig& cfg ) :
	Base( cfg )
	{

	}

	//------------------------------------------------------------------------


	/*virtual*/
	D3D9Buffer::~D3D9Buffer()
	{

	}

	//------------------------------------------------------------------------

	const
	D3D9Buffer::ResourcePtr&
	D3D9Buffer::GetResource() const
	{
		return mResource;
	}

	//------------------------------------------------------------------------

	void
	D3D9Buffer::BindAsVB( IDirect3DDevice9* dev, UINT32 slot ) const
	{
		BindAsVBImpl( dev, slot );
	}

	//------------------------------------------------------------------------

	void
	D3D9Buffer::BindAsIB( IDirect3DDevice9* dev ) const
	{
		BindAsIBImpl( dev );
	}

	//------------------------------------------------------------------------

	void
	D3D9Buffer::SetResource( ResourcePtr::PtrType res )
	{
		mResource.set( res );
	}

	//------------------------------------------------------------------------

	D3D9Buffer::ResourcePtr::PtrType
	D3D9Buffer::GetResourceInternal() const
	{
		return &*mResource;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	D3D9Buffer::BindAsVBImpl( IDirect3DDevice9* dev, UINT32 slot ) const
	{
		dev, slot;
		MD_FERROR( L"Not supported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	D3D9Buffer::BindAsIBImpl( IDirect3DDevice9* dev ) const
	{
		dev;
		MD_FERROR( L"Not supported!" );
	}
	
}