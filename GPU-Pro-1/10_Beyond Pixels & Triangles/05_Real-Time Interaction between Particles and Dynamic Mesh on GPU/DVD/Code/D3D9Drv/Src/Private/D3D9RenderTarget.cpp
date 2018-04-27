#include "Precompiled.h"

#include "Wrap3D/Src/RenderTargetConfig.h"


#include "D3D9Texture1D.h"
#include "D3D9Texture2D.h"
#include "D3D9TextureCube.h"

#include "D3D9RenderTarget.h"

namespace Mod
{
	namespace
	{
		IDirect3DSurface9* extractSurface( D3D9Texture1D& tex )
		{
			IDirect3DSurface9* surf;
			MD_D3DV( static_cast<IDirect3DTexture9*>(&*tex.GetResource())->GetSurfaceLevel( 0, &surf ) );

			return surf;
		}

		IDirect3DSurface9* extractSurface( D3D9TextureCube& tex, UINT32 face )
		{
			IDirect3DSurface9* surf;
			MD_D3DV( static_cast<IDirect3DCubeTexture9*>(&*tex.GetResource())->GetCubeMapSurface( D3DCUBEMAP_FACES(face), 0, &surf ) );

			return surf;
		}

	}

	D3D9RenderTarget::D3D9RenderTarget( const RenderTargetConfig& cfg, IDirect3DDevice9* dev ) :
	Parent( cfg )
	{
		dev;

		// no arrays in D3D9 'n we don't want to write questionable emulations
		MD_FERROR_ON_FALSE( cfg.arrayCount == 1 || !cfg.arrayCount );

		// TODO : Implement buffers as 1D texture in D3D9
		MD_FERROR_ON_TRUE( cfg.buf );

		D3DRESOURCETYPE GetType();

		if( D3D9Texture1D* tex = dynamic_cast< D3D9Texture1D*> ( &*cfg.tex ) )
		{
			mRTSurface.set( extractSurface( static_cast<D3D9Texture1D&>( *cfg.tex ) ) );
		}
		else
		{
			switch( static_cast<D3D9Texture&>( *cfg.tex ).GetResource()->GetType() )
			{
			case D3DRTYPE_TEXTURE:
				mRTSurface = static_cast<D3D9Texture2D&>( *cfg.tex ).GetRenderTargetSurface();
				break;
			case D3DRTYPE_CUBETEXTURE:
				MD_FERROR_ON_FALSE( cfg.arrayCount );
				mRTSurface.set( extractSurface( static_cast<D3D9TextureCube&>( *cfg.tex ), cfg.arrayStart ) );
				break;					

			default:
				MD_FERROR( L"Unsupported resource type!" );
			}
		}

	}

	//------------------------------------------------------------------------
	/*explicit*/

	D3D9RenderTarget::D3D9RenderTarget( IDirect3DSurface9* surf ) :
	Parent( RenderTargetConfig() )
	{
		mRTSurface.set( surf );
	}

	//------------------------------------------------------------------------

	D3D9RenderTarget::~D3D9RenderTarget()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D9RenderTarget::Clear( IDirect3DDevice9* dev, const Math::float4& colr )
	{
		BYTE a = (BYTE)std::min( UINT32( colr.a * 255), UINT32(255) );
		BYTE r = (BYTE)std::min( UINT32( colr.r * 255), UINT32(255) );
		BYTE g = (BYTE)std::min( UINT32( colr.g * 255), UINT32(255) );
		BYTE b = (BYTE)std::min( UINT32( colr.b * 255), UINT32(255) );

		D3DCOLOR d3dcolr = D3DCOLOR_ARGB( a, r, g, b );

		IDirect3DSurface9* zeroSurf( NULL );

		UINT32 i = 0;
		for( ; i < D3D_MAX_SIMULTANEOUS_RENDERTARGETS; i ++ )
		{
			IDirect3DSurface9* rt;
			MD_D3DV( dev->GetRenderTarget( i, &rt ) );

			if( rt )
				rt->Release();

			if( !i ) zeroSurf = rt;

			if( rt == &*mRTSurface )
				break;
		}

		bool restore = false;
		if( i == D3D_MAX_SIMULTANEOUS_RENDERTARGETS )
		{
			MD_D3DV( dev->SetRenderTarget( 0, &*mRTSurface ) );
			restore = true;
			i = 0;
		}

		MD_D3DV( dev->Clear( i, NULL, D3DCLEAR_TARGET, d3dcolr, 1.f, 0 ) );

		if( restore )
		{
			MD_D3DV( dev->SetRenderTarget( 0, zeroSurf ) );
		}


	}

	//------------------------------------------------------------------------

	void
	D3D9RenderTarget::BindTo( IDirect3DDevice9* dev, UINT32 slot ) const
	{
		dev->SetRenderTarget( slot, &*mRTSurface );

		if( GetConfig().tex )
		{
			static_cast<D3D9Texture&>(*GetConfig().tex).Dirty();
		}
	}

	//------------------------------------------------------------------------
	/*static*/

	void
	D3D9RenderTarget::SetBindToZero( IDirect3DDevice9* dev, UINT32 slot  )
	{
		dev->SetRenderTarget( slot, NULL );
	}


}