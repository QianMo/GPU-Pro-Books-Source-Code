#include "Precompiled.h"

#include "Wrap3D/Src/DepthStencilConfig.h"

#include "D3D9Texture2D.h"
#include "D3D9DepthStencil.h"

namespace Mod
{
	D3D9DepthStencil::D3D9DepthStencil( const DepthStencilConfig& cfg, IDirect3DDevice9* dev ) :
	Parent( cfg )
	{
		dev;
		MD_CHECK_TYPE( D3D9Texture2D, &*cfg.tex );

		IDirect3DSurface9* surf;
		MD_D3DV( static_cast< IDirect3DTexture9* >( &*static_cast< D3D9Texture2D& >( *cfg.tex ).GetResource() )->GetSurfaceLevel( 0, &surf ) );

		mDSResource.set( surf );
	}

	//------------------------------------------------------------------------

	/*explicit*/
	D3D9DepthStencil::D3D9DepthStencil( IDirect3DSurface9* surf ) :
	Parent( DepthStencilConfig() )
	{
		mDSResource.set( surf );
	}

	//------------------------------------------------------------------------

	D3D9DepthStencil::~D3D9DepthStencil()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D9DepthStencil::Clear( IDirect3DDevice9* dev, float depthVal, UINT32 stencilVal )
	{
		IDirect3DSurface9* currDS( NULL );

		HRESULT hr = dev->GetDepthStencilSurface( &currDS );

		if( currDS )
		{
			currDS->Release();
		}

		MD_FERROR_ON_TRUE( hr != D3D_OK && hr != D3DERR_NOTFOUND );

		if( currDS != &*mDSResource )
		{
			MD_D3DV( dev->SetDepthStencilSurface( &*mDSResource ) );
		}

		MD_D3DV( dev->Clear( 0, NULL, D3DCLEAR_STENCIL | D3DCLEAR_ZBUFFER, 0, depthVal, stencilVal ) );

		if( currDS != &*mDSResource )
		{
			MD_D3DV( dev->SetDepthStencilSurface( currDS ) );
		}
	}

	//------------------------------------------------------------------------

	void D3D9DepthStencil::BindTo( IDirect3DDevice9* dev ) const
	{
		dev->SetDepthStencilSurface( &*mDSResource );
	}

	//------------------------------------------------------------------------
	/*static*/
	
	void
	D3D9DepthStencil::SetBindToZero( IDirect3DDevice9* dev )
	{
		dev->SetDepthStencilSurface( NULL );
	}
}