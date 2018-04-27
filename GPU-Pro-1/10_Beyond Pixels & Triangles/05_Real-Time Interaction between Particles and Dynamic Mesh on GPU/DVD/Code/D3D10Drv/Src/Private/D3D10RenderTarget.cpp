#include "Precompiled.h"

#include "Wrap3D/Src/BufferConfig.h"
#include "Wrap3D/Src/RenderTargetConfig.h"

#include "D3D10Exception.h"
#include "D3D10Format.h"
#include "D3D10Texture.h"
#include "D3D10Buffer.h"

#include "D3D10RenderTarget.h"

namespace Mod
{
	//------------------------------------------------------------------------

	namespace
	{


		template <typename T>
		struct TextureToRT;

		template <>
		struct TextureToRT<ID3D10Texture1D>
		{
			typedef D3D10_TEXTURE1D_DESC TexDescType;
			static void FillSpecifics( const TexDescType& /*srcdesc*/, D3D10_RENDER_TARGET_VIEW_DESC& desc, const RenderTargetConfig& /*cfg*/ )
			{
				desc.ViewDimension			= D3D10_RTV_DIMENSION_TEXTURE1D;
				desc.Texture1D.MipSlice		= 0;
			}
		};


		template <>
		struct TextureToRT<ID3D10Texture2D>
		{
			typedef D3D10_TEXTURE2D_DESC TexDescType;
			static void FillSpecifics( const TexDescType& srcdesc, D3D10_RENDER_TARGET_VIEW_DESC& desc, const RenderTargetConfig& /*cfg*/ )
			{
				if( srcdesc.SampleDesc.Count > 1 )
				{
					desc.ViewDimension			= D3D10_RTV_DIMENSION_TEXTURE2DMS;
				}
				else
				{
					desc.ViewDimension			= D3D10_RTV_DIMENSION_TEXTURE2D;
					desc.Texture2D.MipSlice		= 0;
				}
			}
		};

		template <>
		struct TextureToRT<ID3D10Texture3D>
		{
			typedef D3D10_TEXTURE3D_DESC TexDescType;
			static void FillSpecifics( const TexDescType& /*srcdesc*/, D3D10_RENDER_TARGET_VIEW_DESC& desc, const RenderTargetConfig& cfg )
			{
				desc.ViewDimension			= D3D10_RTV_DIMENSION_TEXTURE3D;
				desc.Texture3D.MipSlice		= 0;
				desc.Texture3D.FirstWSlice	= cfg.arrayStart;
				desc.Texture3D.WSize		= cfg.arrayCount ? cfg.arrayCount : UINT(-1);
			}
		};

		template<typename Tex>
		void mirrorRTViewDesc( D3D10_RENDER_TARGET_VIEW_DESC &desc, Tex* tex, const RenderTargetConfig& cfg )
		{
			typedef TextureToRT<Tex> RT;

			RT::TexDescType tdesc;
			tex->GetDesc( &tdesc );

			desc.Format	= tdesc.Format;

			RT::FillSpecifics( tdesc, desc, cfg );
		}

		void mirrorRTViewDesc( D3D10_RENDER_TARGET_VIEW_DESC &desc, const Format* fmt, ID3D10Buffer* buf, const RenderTargetConfig& /*cfg*/ )
		{
			D3D10_BUFFER_DESC bdesc;

			buf->GetDesc( &bdesc );

			desc.ViewDimension			= D3D10_RTV_DIMENSION_BUFFER;
			desc.Buffer.ElementOffset	= 0;
			desc.Buffer.ElementWidth	= bdesc.ByteWidth / GetByteCountChecked( fmt );
			desc.Format					= static_cast<const D3D10Format*>(fmt)->GetValue();
		}

		void mirrorRTViewDescCubeMap( D3D10_RENDER_TARGET_VIEW_DESC &desc, ID3D10Texture2D* tex, const RenderTargetConfig& cfg )
		{
			mirrorRTViewDesc( desc, tex, cfg );

			desc.ViewDimension					= D3D10_RTV_DIMENSION_TEXTURE2DARRAY;
			desc.Texture2DArray.FirstArraySlice	= cfg.arrayStart;
			desc.Texture2DArray.ArraySize		= cfg.arrayCount ? cfg.arrayCount : 6;
			desc.Texture2DArray.MipSlice		= 0;

			MD_FERROR_ON_FALSE( cfg.arrayCount + cfg.arrayCount <= 6 );
		}

	}
	
	D3D10RenderTarget::D3D10RenderTarget( const RenderTargetConfig& cfg, ID3D10Device* dev ) :
	Parent( cfg )
	{
		// create render target view
		{

			ResourcePtr res;
			D3D10_RENDER_TARGET_VIEW_DESC desc;

			if( cfg.tex )
			{
				res = static_cast<D3D10Texture*>(cfg.tex.get())->GetResource();

				D3D10_RESOURCE_DIMENSION dim;
				res->GetType( &dim );

				switch( dim )
				{
				case D3D10_RESOURCE_DIMENSION_TEXTURE1D:
					mirrorRTViewDesc( desc, static_cast<ID3D10Texture1D*>( &*res ), cfg );
					break;

				case D3D10_RESOURCE_DIMENSION_TEXTURE2D:
					{
						ID3D10Texture2D* tex = static_cast<ID3D10Texture2D*>( &*res );
						D3D10_TEXTURE2D_DESC t2desc;
						tex->GetDesc( &t2desc );

						if( t2desc.MiscFlags & D3D10_RESOURCE_MISC_TEXTURECUBE )
							mirrorRTViewDescCubeMap( desc, tex, cfg );
						else
							mirrorRTViewDesc( desc, tex, cfg );
					}
					break;

				case D3D10_RESOURCE_DIMENSION_TEXTURE3D:
					mirrorRTViewDesc( desc, static_cast<ID3D10Texture3D*>( &*res ), cfg );
					break;

				default:
					MD_FERROR( L"Resource dimenstion type not supported yet!");
				}
			}
			else
			{
				ID3D10Buffer* buf = &*static_cast<D3D10Buffer&>(*cfg.buf).GetResource();
				
				res.set( buf );
				res->AddRef();

				mirrorRTViewDesc( desc, cfg.buf->GetConfig().GetFormat(), buf, cfg );
			}


			ID3D10RenderTargetView* rtv;
			
			D3D10_THROW_IF( dev->CreateRenderTargetView( &*res, &desc, &rtv ) );

			mRTView.set( rtv ); 
		}
	}

	//------------------------------------------------------------------------

	D3D10RenderTarget::~D3D10RenderTarget()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D10RenderTarget::Clear( ID3D10Device* dev, const Math::float4& colr )
	{
		dev->ClearRenderTargetView( &*mRTView, colr.elems );
	}

	//------------------------------------------------------------------------

	void
	D3D10RenderTarget::BindTo( BindType& target ) const
	{
		target = &*mRTView;
	}

	//------------------------------------------------------------------------

	void
	D3D10RenderTarget::SetBindToZero( BindType& target )
	{
		target = NULL;
	}

}