#include "Precompiled.h"

#include "Wrap3D/Src/DepthStencilConfig.h"

#include "D3D10DepthStencil.h"
#include "D3D10Exception.h"
#include "D3D10Format.h"
#include "D3D10Texture.h"

namespace Mod
{

	namespace
	{
		template <typename T>
		struct TextureToDS;

		template <>
		struct TextureToDS<ID3D10Texture1D>
		{
			typedef D3D10_TEXTURE1D_DESC TexDescType;
			static void FillSpecifics( const TexDescType& /*srcdesc*/, D3D10_DEPTH_STENCIL_VIEW_DESC& desc, const DepthStencilConfig& /*cfg*/ )
			{
				desc.ViewDimension			= D3D10_DSV_DIMENSION_TEXTURE1D;
				desc.Texture1D.MipSlice		= 0;
			}
		};


		template <>
		struct TextureToDS<ID3D10Texture2D>
		{
			typedef D3D10_TEXTURE2D_DESC TexDescType;
			static void FillSpecifics( const TexDescType& srcdesc, D3D10_DEPTH_STENCIL_VIEW_DESC& desc, const DepthStencilConfig& cfg )
			{
				UINT arrSlice	=  cfg.arrayStart;
				UINT arrCount	= cfg.arrayCount ? cfg.arrayCount : srcdesc.ArraySize;

				if( srcdesc.ArraySize > 1 )
				{
					if( srcdesc.SampleDesc.Count > 1 )
					{
						desc.ViewDimension						= D3D10_DSV_DIMENSION_TEXTURE2DMSARRAY;
						desc.Texture2DMSArray.ArraySize			= arrCount;
						desc.Texture2DMSArray.FirstArraySlice	= arrSlice;
					}
					else
					{
						desc.ViewDimension						= D3D10_DSV_DIMENSION_TEXTURE2DARRAY;
						desc.Texture2DArray.ArraySize			= arrCount;
						desc.Texture2DArray.FirstArraySlice		= arrSlice;
						desc.Texture2DArray.MipSlice			= 0;
					}
				}
				else
				{
					MD_FERROR_ON_FALSE( arrCount == 1 && !arrSlice );
					if( srcdesc.SampleDesc.Count > 1 )
					{
						desc.ViewDimension			= D3D10_DSV_DIMENSION_TEXTURE2DMS;
					}
					else
					{
						desc.ViewDimension			= D3D10_DSV_DIMENSION_TEXTURE2D;
						desc.Texture2D.MipSlice		= 0;
					}
				}
			}
		};


		template<typename Tex>
		void mirrorDSViewDesc( D3D10_DEPTH_STENCIL_VIEW_DESC &desc, Tex* tex, const DepthStencilConfig& cfg )
		{
			typedef TextureToDS<Tex> RT;

			RT::TexDescType tdesc;
			tex->GetDesc( &tdesc );

			desc.Format	= tdesc.Format;

			RT::FillSpecifics( tdesc, desc, cfg );
		}
	}

	//------------------------------------------------------------------------


	D3D10DepthStencil::D3D10DepthStencil( const DepthStencilConfig& cfg, ID3D10Device* dev ) :
	Parent( cfg )
	{

		ResourcePtr res = static_cast<D3D10Texture*>(cfg.tex.get())->GetResource();
		
		D3D10_DEPTH_STENCIL_VIEW_DESC dsvDesc;

		D3D10_RESOURCE_DIMENSION dim;
		res->GetType( &dim );

		switch( dim )
		{
		case D3D10_RESOURCE_DIMENSION_TEXTURE1D:
			mirrorDSViewDesc( dsvDesc, static_cast<ID3D10Texture1D*>(&*res), cfg );
			break;
		case D3D10_RESOURCE_DIMENSION_TEXTURE2D:
			mirrorDSViewDesc( dsvDesc, static_cast<ID3D10Texture2D*>(&*res), cfg );
			break;
		default:
			MD_FERROR( L"Unsupported resource dimension!");
		}

		// override format like ppl want
		dsvDesc.Format = static_cast<const D3D10Format*>(cfg.fmt)->GetValue();

		ID3D10DepthStencilView* dsv;
		D3D10_THROW_IF( dev->CreateDepthStencilView( &*res, &dsvDesc, &dsv ) );

		mDSView.set( dsv );
	}

	//------------------------------------------------------------------------


	D3D10DepthStencil::~D3D10DepthStencil()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D10DepthStencil::Clear( ID3D10Device* dev, float depthVal, UINT32 stencilVal )
	{
		dev->ClearDepthStencilView( &*mDSView, D3D10_CLEAR_DEPTH | D3D10_CLEAR_STENCIL, depthVal, (UINT8)stencilVal );
	}

	//------------------------------------------------------------------------
	void
	D3D10DepthStencil::BindTo( ID3D10DepthStencilView*& ds ) const
	{
		ds = &*mDSView;
	}

}