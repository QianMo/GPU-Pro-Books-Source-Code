#include "Precompiled.h"

#include "Wrap3D/Src/TextureConfig.h"

#include "D3D10Texture.h"
#include "D3D10SRView.h"

#include "D3D10Texture1D.h"
#include "D3D10Texture2D.h"
#include "D3D10Texture3D.h"

#include "D3D10FormatMap.h"
#include "D3D10UsageMap.h"

namespace Mod
{

	D3D10Texture::D3D10Texture( const TextureConfig& cfg ) :
	Texture( cfg )
	{

	}

	//------------------------------------------------------------------------

	D3D10Texture::~D3D10Texture()
	{

	}

	//------------------------------------------------------------------------
	
	D3D10Texture::D3D10Texture( const TextureConfig& cfg, ResourcePtr res ):
	Base( cfg ),
	mResource( res )
	{

	}

	//------------------------------------------------------------------------

	const D3D10Texture::ResourcePtr&
	D3D10Texture::GetResource() const
	{
		return mResource;
	}

	//------------------------------------------------------------------------
		
	void
	D3D10Texture::SetResource( ResourcePtr::PtrType res )
	{
		MD_FERROR_ON_FALSE( res );		
		mResource.set( res );
	}

	//------------------------------------------------------------------------

	ID3D10Device*
	D3D10Texture::GetD3D10Device() const
	{
		ID3D10Device* res;
		mResource->GetDevice(&res);
		return res;
	}

	//------------------------------------------------------------------------

	namespace
	{

		template <typename T, typename U>
		void FillCommon( T& targ, const U& src, const D3D10FormatMap& fm, const D3D10UsageMap& um )
		{

			targ.fmt		= fm.GetFormat(src.Format);
			MD_FERROR_ON_FALSE( targ.fmt );

			targ.usage		= um.GetUsage(src.Usage);
			MD_FERROR_ON_FALSE( targ.usage );

			targ.numMips	= src.MipLevels;

			if( src.BindFlags & D3D10_BIND_RENDER_TARGET )
				targ.renderTarget = true;
			else
				targ.renderTarget = false;

			if( src.BindFlags & D3D10_BIND_SHADER_RESOURCE)
				targ.shaderResource = true;
			else
				targ.shaderResource = false;

			if( src.BindFlags & D3D10_BIND_DEPTH_STENCIL )
				targ.depthStencil = true;
			else
				targ.depthStencil = false;

		}

		void FillConfig( Texture3DConfig& cfg, D3D10Texture::ResourcePtr res, const D3D10FormatMap& fm, const D3D10UsageMap& um )
		{
			D3D10_TEXTURE3D_DESC desc;
			static_cast<ID3D10Texture3D*>(&*res)->GetDesc( &desc );

			cfg.width	= desc.Width;
			cfg.height	= desc.Height;
			cfg.depth	= desc.Depth;

			FillCommon( cfg, desc, fm, um );
		}

		void FillConfig( Texture2DConfig& cfg, D3D10Texture::ResourcePtr res, const D3D10FormatMap& fm, const D3D10UsageMap& um )
		{
			D3D10_TEXTURE2D_DESC desc;
			static_cast<ID3D10Texture2D*>(&*res)->GetDesc( &desc );

			cfg.width	= desc.Width;
			cfg.height	= desc.Height;

			FillCommon( cfg, desc, fm, um );
		}

		void FillConfig( Texture1DConfig& cfg, D3D10Texture::ResourcePtr res, const D3D10FormatMap& fm, const D3D10UsageMap& um )
		{
			D3D10_TEXTURE1D_DESC desc;
			static_cast<ID3D10Texture1D*>(&*res)->GetDesc( &desc );

			cfg.length	= desc.Width;

			FillCommon( cfg, desc, fm, um );
		}
	}

	
	TexturePtr CreateTextureFromResource( D3D10Texture::ResourcePtr res, const D3D10FormatMap& fm, const D3D10UsageMap& um )
	{
		D3D10_RESOURCE_DIMENSION type;
		res->GetType(&type);

		switch( type )
		{
		case D3D10_RESOURCE_DIMENSION_TEXTURE1D:
			{
				Texture1DConfig cfg;
				FillConfig( cfg, res, fm, um );
				return TexturePtr( new D3D10Texture1D( cfg, res ) );
			}
			break;
		case D3D10_RESOURCE_DIMENSION_TEXTURE2D:
			{
				Texture2DConfig cfg;
				FillConfig( cfg, res, fm, um );

				Texture2DConfigEx ecfg(cfg);

				D3D10_TEXTURE2D_DESC desc;
				static_cast<ID3D10Texture2D*>(&*res)->GetDesc( &desc );

				if( desc.MiscFlags & D3D10_RESOURCE_MISC_TEXTURECUBE )
				{
					ecfg.cubeTexture = true;
					return TexturePtr( new D3D10Texture2D( ecfg, res ) );
				}
				else
				{
					return TexturePtr( new D3D10Texture2D( ecfg, res ) );
				}
			}
			break;
		case D3D10_RESOURCE_DIMENSION_TEXTURE3D:
			{
				Texture3DConfig cfg;
				FillConfig( cfg, res, fm, um );
				return TexturePtr( new D3D10Texture3D( cfg, res ) );
			}
			break;

		default:
			MD_FERROR( L"Resource dimmension not supported!" );
		}

		return TexturePtr();
	}

}