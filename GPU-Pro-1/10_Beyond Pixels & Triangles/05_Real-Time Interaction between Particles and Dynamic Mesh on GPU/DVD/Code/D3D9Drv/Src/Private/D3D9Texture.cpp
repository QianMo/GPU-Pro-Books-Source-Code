#include "Precompiled.h"

#include "Wrap3D/Src/TextureConfig.h"
#include "Wrap3D/Src/Usages.h"

#include "D3D9Helpers/Src/D3D9FormatMap.h"
#include "D3D9Device.h"
#include "D3D9Texture.h"

#include "D3D9TextureCube.h"
#include "D3D9Texture2D.h"
#include "D3D9Texture3D.h"

namespace Mod
{

	D3D9Texture::D3D9Texture( const TextureConfig& cfg ) :
	Base( cfg )
	{

	}

	//------------------------------------------------------------------------

	D3D9Texture::~D3D9Texture()
	{

	}

	//------------------------------------------------------------------------
	
	D3D9Texture::D3D9Texture( const TextureConfig& cfg, ResourcePtr res ):
	Base( cfg ),
	mResource( res )
	{

	}

	//------------------------------------------------------------------------

	const D3D9Texture::ResourcePtr&
	D3D9Texture::GetResource() const
	{
		return mResource;
	}

	//------------------------------------------------------------------------

	void
	D3D9Texture::Dirty()
	{
		DirtyImpl();
	}

	//------------------------------------------------------------------------

	void
	D3D9Texture::CopyTo( IDirect3DDevice9* dev, D3D9Texture& dest )
	{
		CopyToImpl( dev, dest );
	}

	//------------------------------------------------------------------------

	void
	D3D9Texture::BindTo( IDirect3DDevice9* dev, UINT32 slot )
	{
		UpdateFromRTImpl( dev );
		dev->SetTexture( slot, &*mResource );
	}

	//------------------------------------------------------------------------

	void
	D3D9Texture::BindTo( ID3DXEffect* eff, D3DXHANDLE handle )
	{		
		IDirect3DDevice9* dev;
		MD_D3DV( eff->GetDevice( &dev ) );
		dev->Release();

		UpdateFromRTImpl( dev );
		MD_D3DV( eff->SetTexture( handle, &*mResource ) );
	}

	//------------------------------------------------------------------------
		
	void
	D3D9Texture::SetResource( ResourcePtr::PtrType res )
	{
		MD_FERROR_ON_FALSE( res );		
		mResource.set( res );
	}

	//------------------------------------------------------------------------

	IDirect3DDevice9*
	D3D9Texture::GetD3D9Device() const
	{
		IDirect3DDevice9* res;
		MD_D3DV( mResource->GetDevice(&res) );
		res->Release();

		return res;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9Texture::DirtyImpl()
	{
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9Texture::CopyToImpl( IDirect3DDevice9* dev, D3D9Texture& dest )
	{
		dev, dest;
		MD_FERROR( L"Unsupported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9Texture::UpdateFromRTImpl( IDirect3DDevice9* dev )
	{
		dev;
	}

	//------------------------------------------------------------------------

	namespace
	{
		void FillConfig( TextureConfig& oCfg, D3DFORMAT fmt, IDirect3DBaseTexture9& tex, const D3D9Device& dev )
		{
			oCfg.usage				= dev.GetUsages().DEFAULT;
			oCfg.numMips			= tex.GetLevelCount();
			oCfg.fmt				= dev.GetD3D9FormatMap().GetFormat( fmt );
		}
	}

	TexturePtr CreateTextureFromResource( D3D9Texture::ResourcePtr res, D3D9Device& dev )
	{
		TexturePtr result;

		D3DRESOURCETYPE type = res->GetType();

		switch( type )
		{
		case D3DRTYPE_TEXTURE:
			{
				D3DSURFACE_DESC desc;
				static_cast<IDirect3DTexture9&>(*res).GetLevelDesc( 0, &desc );

				Texture2DConfig cfg;
				FillConfig( cfg, desc.Format, *res, dev );

				cfg.width	= desc.Width;
				cfg.height	= desc.Height;

				result.reset( new D3D9Texture2D( cfg, res ) );
			}
			break;
		case D3DRTYPE_CUBETEXTURE:
			{
				D3DSURFACE_DESC desc;
				static_cast<IDirect3DCubeTexture9&>(*res).GetLevelDesc( 0, &desc );

				TextureCUBEConfig cfg;
				FillConfig( cfg, desc.Format, *res, dev );

				cfg.width			= desc.Width;
				cfg.height			= desc.Height;

				result.reset( new D3D9TextureCube( cfg, res ) );
			}
			break;
		case D3DRTYPE_VOLUMETEXTURE:
			{
				D3DVOLUME_DESC desc;
				static_cast<IDirect3DVolumeTexture9&>(*res).GetLevelDesc( 0, &desc );

				Texture3DConfig cfg;
				FillConfig( cfg, desc.Format, *res, dev );

				cfg.width			= desc.Width;
				cfg.height			= desc.Height;
				cfg.depth			= desc.Depth;		

				result.reset( new D3D9Texture3D( cfg, res ) );
			}
			break;

		default:
			MD_FERROR( L"Resource dimmension not supported!" );
		}

		return result;
	}


}