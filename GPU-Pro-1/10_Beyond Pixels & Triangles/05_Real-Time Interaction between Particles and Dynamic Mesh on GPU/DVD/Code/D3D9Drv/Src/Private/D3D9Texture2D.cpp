#include "Precompiled.h"

#include "Wrap3D/Src/TextureConfig.h"

#include "D3D9Usage.h"
#include "D3D9Format.h"

#include "D3D9Texture2D.h"

namespace Mod
{
	D3D9Texture2D::D3D9Texture2D( const Texture2DConfig& cfg, IDirect3DDevice9* dev ) :
	Parent( cfg ),
	mDirty( true )
	{
		MD_FERROR_ON_TRUE( cfg.sampleCount > 1 && !cfg.renderTarget );

		const D3D9Usage* usg = static_cast<const D3D9Usage*>( cfg.usage );
		IDirect3DTexture9* tex;
		MD_D3DV( dev->CreateTexture( cfg.width, cfg.height, cfg.numMips, usg->GetConfig().textureUsage, static_cast< const D3D9Format*>(cfg.fmt)->GetValue(), usg->GetConfig().texPool, &tex, NULL ) );
		SetResource( tex );

		if( cfg.renderTarget )
		{
			if( cfg.sampleCount > 1 )
			{
				IDirect3DSurface9* surf;
				MD_D3DV( dev->CreateRenderTarget( cfg.width, cfg.height, static_cast< const D3D9Format*>(cfg.fmt)->GetValue(), D3DMULTISAMPLE_TYPE( cfg.sampleCount > 1 ? cfg.sampleCount : 0 ), 0, FALSE, &surf, NULL ) );
				mRenderTargetSurface.set( surf );

				MD_D3DV( tex->GetSurfaceLevel( 0, &surf ) );
				mResolveSurface.set( surf );
			}
			else
			{
				IDirect3DSurface9* surf;
				MD_D3DV( tex->GetSurfaceLevel( 0, &surf ) );
				mRenderTargetSurface.set( surf );
			}
		}
	}

	//------------------------------------------------------------------------

	D3D9Texture2D::D3D9Texture2D( const Texture2DConfig& cfg, ResourcePtr res ) :
	Parent( cfg, res )
	{

	}

	//------------------------------------------------------------------------

	D3D9Texture2D::~D3D9Texture2D()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D9Texture2D::ResolveTo( IDirect3DDevice9* dev, D3D9Texture2D& tex )
	{
		MD_FERROR_ON_FALSE( tex.GetConfig().renderTarget && static_cast<const Texture2DConfig&>(tex.GetConfig()).sampleCount <= 1 );

		MD_FERROR_ON_TRUE( mResolveSurface.null() );
		UpdateFromRTImpl( dev );

		MD_D3DV( dev->UpdateSurface( &*mResolveSurface, NULL, &*tex.mRenderTargetSurface, NULL ) );
	}

	//------------------------------------------------------------------------

	const D3D9Texture2D::SurfacePtr&
	D3D9Texture2D::GetRenderTargetSurface() const
	{
		return mRenderTargetSurface;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	D3D9Texture2D::DirtyImpl() /*OVERRIDE*/
	{
		mDirty = true;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void D3D9TexCopyTo2D( IDirect3DDevice9* dev, IDirect3DTexture9& src_t, IDirect3DTexture9& dest_t )
	{
		UINT32 levelCount1 = src_t.GetLevelCount();
		UINT32 levelCount2 = dest_t.GetLevelCount();

		for( UINT32 i = 0, e = std::min( levelCount1, levelCount2); i < e; i ++ )
		{
			IDirect3DSurface9 *src_s, *dest_s;
			MD_D3DV( src_t.GetSurfaceLevel( i, & src_s ) );
			MD_D3DV( dest_t.GetSurfaceLevel( i, & dest_s ) );

			MD_D3DV( dev->UpdateSurface( src_s, NULL, dest_s, NULL ) );
		}
	}
	
	void
	D3D9Texture2D::CopyToImpl( IDirect3DDevice9* dev, D3D9Texture& dest ) /*OVERRIDE*/
	{
		IDirect3DTexture9& src_t	= static_cast< IDirect3DTexture9& >( *GetResource() );
		IDirect3DTexture9& dest_t	= static_cast< IDirect3DTexture9& >( *dest.GetResource() );

		UpdateFromRTImpl( dev );

		D3D9TexCopyTo2D( dev, src_t, dest_t );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9Texture2D::UpdateFromRTImpl( IDirect3DDevice9* dev ) /*OVERRIDE*/
	{
		if( !mResolveSurface.null() && mDirty )
		{
			MD_D3DV( dev->StretchRect( &*mRenderTargetSurface, NULL, &*mResolveSurface, NULL, D3DTEXF_NONE ) );
		}

		mDirty = false;
	}

}