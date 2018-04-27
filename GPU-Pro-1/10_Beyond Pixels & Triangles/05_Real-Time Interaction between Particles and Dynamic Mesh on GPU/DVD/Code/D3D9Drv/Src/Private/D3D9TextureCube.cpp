#include "Precompiled.h"

#include "Wrap3D/Src/TextureConfig.h"

#include "D3D9Usage.h"
#include "D3D9Format.h"

#include "D3D9TextureCube.h"

namespace Mod
{
	D3D9TextureCube::D3D9TextureCube( const TextureCUBEConfig& cfg, IDirect3DDevice9* dev ) :
	Parent( cfg )
	{
		MD_FERROR_ON_FALSE( cfg.width == cfg.height );
		const D3D9Usage* usg = static_cast<const D3D9Usage*>( cfg.usage );
		IDirect3DCubeTexture9* tex;
		MD_D3DV( dev->CreateCubeTexture( cfg.width, cfg.numMips, usg->GetConfig().textureUsage, static_cast< const D3D9Format*>(cfg.fmt)->GetValue(), usg->GetConfig().texPool, &tex, NULL ) );
		SetResource( tex );
	}

	//------------------------------------------------------------------------

	D3D9TextureCube::D3D9TextureCube( const TextureCUBEConfig& cfg, ResourcePtr res ) :
	Parent( cfg, res )
	{

	}

	//------------------------------------------------------------------------

	D3D9TextureCube::~D3D9TextureCube()
	{

	}

}