#include "Precompiled.h"

#include "Wrap3D/Src/TextureConfig.h"

#include "D3D9Usage.h"
#include "D3D9Format.h"

#include "D3D9Texture3D.h"

namespace Mod
{
	D3D9Texture3D::D3D9Texture3D( const Texture3DConfig& cfg, IDirect3DDevice9* dev ) :
	Parent( cfg )
	{
		const D3D9Usage* usg = static_cast<const D3D9Usage*>( cfg.usage );
		IDirect3DVolumeTexture9* tex;
		MD_D3DV( dev->CreateVolumeTexture( cfg.width, cfg.height, cfg.depth, cfg.numMips, usg->GetConfig().textureUsage, static_cast< const D3D9Format*>(cfg.fmt)->GetValue(), usg->GetConfig().texPool, &tex, NULL ) );
		SetResource( tex );
	}

	//------------------------------------------------------------------------

	D3D9Texture3D::D3D9Texture3D( const Texture3DConfig& cfg, ResourcePtr res ) :
	Parent( cfg, res )
	{

	}

	//------------------------------------------------------------------------

	D3D9Texture3D::~D3D9Texture3D()
	{

	}

}