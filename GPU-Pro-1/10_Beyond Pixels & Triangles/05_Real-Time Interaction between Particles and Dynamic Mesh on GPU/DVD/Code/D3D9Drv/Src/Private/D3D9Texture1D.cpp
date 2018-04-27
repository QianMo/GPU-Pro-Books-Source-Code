#include "Precompiled.h"

#include "Wrap3D/Src/TextureConfig.h"

#include "D3D9Usage.h"
#include "D3D9Format.h"

#include "D3D9Texture1D.h"

namespace Mod
{
	D3D9Texture1D::D3D9Texture1D( const Texture1DConfig& cfg, IDirect3DDevice9* dev ) :
	Parent( cfg )
	{
		const D3D9Usage* usg = static_cast<const D3D9Usage*>( cfg.usage );
		IDirect3DTexture9* tex;
		MD_D3DV( dev->CreateTexture( cfg.length, 1, cfg.numMips, usg->GetConfig().textureUsage, static_cast< const D3D9Format*>(cfg.fmt)->GetValue(), usg->GetConfig().texPool, &tex, NULL ) );
		SetResource( tex );
	}

	//------------------------------------------------------------------------

	D3D9Texture1D::D3D9Texture1D( const Texture1DConfig& cfg, ResourcePtr res ) :
	Parent( cfg, res )
	{

	}

	//------------------------------------------------------------------------

	D3D9Texture1D::~D3D9Texture1D()
	{

	}

	//------------------------------------------------------------------------
	/*virtual*/

	void D3D9TexCopyTo2D( IDirect3DDevice9* dev, IDirect3DTexture9& src_t, IDirect3DTexture9& dest_t );
	
	void
	D3D9Texture1D::CopyToImpl( IDirect3DDevice9* dev, D3D9Texture& dest ) /*OVERRIDE*/
	{
		IDirect3DTexture9& src_t	= static_cast< IDirect3DTexture9& >( *GetResource() );
		IDirect3DTexture9& dest_t	= static_cast< IDirect3DTexture9& >( *dest.GetResource() );

		D3D9TexCopyTo2D( dev, src_t, dest_t );
	}


}