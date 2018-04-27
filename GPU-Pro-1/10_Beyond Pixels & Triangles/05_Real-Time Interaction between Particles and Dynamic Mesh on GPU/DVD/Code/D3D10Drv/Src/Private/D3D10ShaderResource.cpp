#include "Precompiled.h"

#include "Wrap3D/Src/ShaderResourceConfig.h"
#include "Wrap3D/Src/BufferConfig.h"

#include "D3D10ShaderResource.h"
#include "D3D10Format.h"
#include "D3D10Exception.h"
#include "D3D10Texture.h"
#include "D3D10Buffer.h"

namespace Mod
{

	//------------------------------------------------------------------------

	namespace
	{


		template <typename T>
		struct TextureToSR;

		template <>
		struct TextureToSR<ID3D10Texture1D>
		{
			typedef D3D10_TEXTURE1D_DESC TexDescType;
			static void FillSpecifics( const TexDescType& /*srcdesc*/, D3D10_SHADER_RESOURCE_VIEW_DESC& desc )
			{
				desc.ViewDimension				= D3D10_SRV_DIMENSION_TEXTURE1D;
				desc.Texture1D.MipLevels		= UINT(-1);
				desc.Texture1D.MostDetailedMip	= 0;
			}
		};


		template <>
		struct TextureToSR<ID3D10Texture2D>
		{
			typedef D3D10_TEXTURE2D_DESC TexDescType;
			static void FillSpecifics( const TexDescType& srcdesc, D3D10_SHADER_RESOURCE_VIEW_DESC& desc )
			{
				if( srcdesc.SampleDesc.Count > 1 )
				{
					desc.ViewDimension				= D3D10_SRV_DIMENSION_TEXTURE2DMS;
				}
				else
				{
					desc.ViewDimension				= D3D10_SRV_DIMENSION_TEXTURE2D;
					desc.Texture2D.MipLevels		= UINT(-1);
					desc.Texture2D.MostDetailedMip	= 0;
				}
			}
		};

		template <>
		struct TextureToSR<ID3D10Texture3D>
		{
			typedef D3D10_TEXTURE3D_DESC TexDescType;
			static void FillSpecifics( const TexDescType& /*srcdesc*/, D3D10_SHADER_RESOURCE_VIEW_DESC& desc )
			{
				desc.ViewDimension				= D3D10_SRV_DIMENSION_TEXTURE3D;
				desc.Texture3D.MipLevels		= UINT(-1);
				desc.Texture3D.MostDetailedMip	= 0;
			}
		};

		template<typename Tex>
		void mirrorSRViewDesc( D3D10_SHADER_RESOURCE_VIEW_DESC &desc, Tex* tex )
		{
			typedef TextureToSR<Tex> SR;

			SR::TexDescType tdesc;
			tex->GetDesc( &tdesc );

			desc.Format	= tdesc.Format;

			SR::FillSpecifics( tdesc, desc );
		}

		void mirrorSRViewDesc( D3D10_SHADER_RESOURCE_VIEW_DESC &desc, ID3D10Buffer* /*buf*/ )
		{
			desc.ViewDimension				= D3D10_SRV_DIMENSION_BUFFER;
			desc.Buffer.ElementOffset		= 0;
		}

		void mirrorSRViewDescCube( D3D10_SHADER_RESOURCE_VIEW_DESC &desc, ID3D10Texture2D* /*t2d*/ )
		{
			desc.ViewDimension					= D3D10_SRV_DIMENSION_TEXTURECUBE;
			desc.TextureCube.MipLevels			= UINT(-1);
			desc.TextureCube.MostDetailedMip	= 0;
		}
	}


	//------------------------------------------------------------------------

	D3D10ShaderResource::D3D10ShaderResource( const ShaderResourceConfig& cfg, ID3D10Device* dev ) :
	Parent( cfg )
	{
		// create render target view
		{
			ResourcePtr res;
			if( cfg.tex )
			{
				res = static_cast<D3D10Texture*>(cfg.tex.get())->GetResource();
			}
			else
			{
				ID3D10Buffer* d3dres = &*static_cast<D3D10Buffer*>(cfg.buf.get())->GetResource();
				d3dres->AddRef();
				res.set( d3dres );
			}

			D3D10_RESOURCE_DIMENSION dim;
			res->GetType( &dim );

			D3D10_SHADER_RESOURCE_VIEW_DESC desc;

			switch( dim )
			{
			case D3D10_RESOURCE_DIMENSION_TEXTURE1D:
				mirrorSRViewDesc( desc, static_cast<ID3D10Texture1D*>( &*res ) );
				break;

			case D3D10_RESOURCE_DIMENSION_TEXTURE2D:
				{
					ID3D10Texture2D* t2d = static_cast<ID3D10Texture2D*>( &*res );
					D3D10_TEXTURE2D_DESC tdesc;
					t2d->GetDesc( &tdesc );

					if( tdesc.MiscFlags & D3D10_RESOURCE_MISC_TEXTURECUBE )
						mirrorSRViewDescCube( desc, t2d );
					else
						mirrorSRViewDesc( desc, t2d );
				}
				break;

			case D3D10_RESOURCE_DIMENSION_TEXTURE3D:
				mirrorSRViewDesc( desc, static_cast<ID3D10Texture3D*>( &*res ) );
				break;

			case D3D10_RESOURCE_DIMENSION_BUFFER:
				{
					const D3D10Format* fmt = static_cast<const D3D10Format*>(cfg.buf->GetConfig().GetFormat());
					mirrorSRViewDesc( desc, static_cast<ID3D10Buffer*>( &*res ));
					desc.Format					= fmt->GetValue();
					desc.Buffer.ElementWidth	= static_cast<UINT32>( cfg.buf->GetConfig().GetByteSize() / GetByteCountChecked( fmt ) );
				}
				break;

			default:
				MD_FERROR( L"Resource dimenstion type not supported yet!");

			}

			// override format like ppl want
			desc.Format = static_cast<const D3D10Format*>(cfg.fmt)->GetValue();


			ID3D10ShaderResourceView* srv;
			
			D3D10_THROW_IF( dev->CreateShaderResourceView( &*res, &desc, &srv ) );

			mSRView.set( srv ); 
		}
	}

	//------------------------------------------------------------------------

	D3D10ShaderResource::~D3D10ShaderResource()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D10ShaderResource::BindTo( BindType1* srv ) const
	{
		srv->SetResource( &*mSRView );
	}

	//------------------------------------------------------------------------

	/*static*/
	void
	D3D10ShaderResource::SetBindToZero( BindType1* srv )
	{
		srv->SetResource( NULL );
	}

	//------------------------------------------------------------------------

	void
	D3D10ShaderResource::BindTo( BindType2* srv ) const
	{
		srv = &*mSRView;
	}

	//------------------------------------------------------------------------

	/*static*/
	void
	D3D10ShaderResource::SetBindToZero( BindType2* srv )
	{
		srv = NULL;
	}

}