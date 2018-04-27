#include "Precompiled.h"

#include "D3D9TextureCoordinatorConfig.h"
#include "D3D9TextureCoordinator.h"

#define MD_NAMESPACE D3D9TextureCoordinatorNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{

	//------------------------------------------------------------------------

	D3D9TextureCoordinator::Texture::Texture():
	dirty( true )
	{

	}


	//------------------------------------------------------------------------

	D3D9TextureCoordinator::D3D9TextureCoordinator( const D3D9TextureCoordinatorConfig& cfg ) :
	Parent( cfg ),
	mTextures( cfg.numTextures )
	{
	}

	//------------------------------------------------------------------------

	D3D9TextureCoordinator::~D3D9TextureCoordinator() 
	{
	}

	//------------------------------------------------------------------------

	void
	D3D9TextureCoordinator::SetTexture( UINT32 slot, IDirect3DBaseTexture9* texture )
	{
		if( mTextures[ slot ].tex != texture )		
		{
			mTextures[ slot ].tex.set( texture );
			if( texture )
			{
				mTextures[ slot ].tex->AddRef();
			}
			mTextures[ slot ].dirty = true;
		}
	}

	//------------------------------------------------------------------------

	void
	D3D9TextureCoordinator::Update()
	{
		const ConfigType& cfg = GetConfig();

		for( size_t i = 0, e = mTextures.size(); i < e; i ++ )
		{
			if( mTextures[ i ].dirty )
			{
				cfg.device->SetTexture( (DWORD)i, &*mTextures[ i ].tex );
				mTextures[ i ].dirty = false;
			}
		}
	}

	//------------------------------------------------------------------------

	void
	D3D9TextureCoordinator::Sync()
	{
		const ConfigType& cfg = GetConfig();
		for( size_t i = 0, e = mTextures.size(); i < e; i ++ )
		{
			IDirect3DBaseTexture9* tex;
			MD_D3DV( cfg.device->GetTexture( (DWORD)i, &tex ) );
			SetTexture( (UINT32)i, tex );
			mTextures[ i ].dirty = false;
		}
	}

}