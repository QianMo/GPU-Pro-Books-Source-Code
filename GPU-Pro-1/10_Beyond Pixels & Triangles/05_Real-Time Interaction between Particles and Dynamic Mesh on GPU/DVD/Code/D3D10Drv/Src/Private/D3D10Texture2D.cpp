#include "Precompiled.h"
#include "D3D10Texture2D.h"

#include "Wrap3D\Src\TextureConfig.h"

namespace Mod
{

	//------------------------------------------------------------------------

	Texture2DConfigEx::Texture2DConfigEx( const Texture2DConfig& cfg ) :
	Texture2DConfig( cfg )
	{
		cubeTexture = false;
	}

	//------------------------------------------------------------------------

	Texture2DConfigEx*
	Texture2DConfigEx::Clone() const
	{
		return new Texture2DConfigEx( *this );
	}

	//------------------------------------------------------------------------

	Texture2DConfigEx::Texture2DConfigEx( const TextureCUBEConfig& cfg )
	{
		CopyFrom( cfg );
		cubeTexture = true;
	}

	//------------------------------------------------------------------------

	HRESULT ( MD_D3D_CALLING_CONV ID3D10Device::*D3D10Texture2DConfig::CreateTexture)( const DescType*, const D3D10_SUBRESOURCE_DATA*, ResType** ) = &ID3D10Device::CreateTexture2D;

	//------------------------------------------------------------------------

	D3D10Texture2D::D3D10Texture2D( const TexConfigType& cfg, ID3D10Device* dev ):
	Parent ( cfg, dev )	
	{

	}

	//------------------------------------------------------------------------

	D3D10Texture2D::D3D10Texture2D( const TexConfigType& cfg, ResourcePtr ptr ):
	Parent( cfg, ptr )
	{
		
	}

	//------------------------------------------------------------------------

	D3D10Texture2D::~D3D10Texture2D()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D10Texture2D::Configure( Parent::DescType& desc, const TexConfigType& cfg )
	{
		MD_FERROR_ON_FALSE( !cfg.cubeTexture || cfg.width == cfg.height );
		MD_FERROR_ON_FALSE( !cfg.cubeTexture || cfg.arraySize == 1 );

		desc.ArraySize	= cfg.cubeTexture ? 6 : cfg.arraySize;
		desc.Width		= cfg.width;
		desc.Height		= cfg.height;

		if( cfg.cubeTexture )
			desc.MiscFlags	|= D3D10_RESOURCE_MISC_TEXTURECUBE;

		desc.SampleDesc.Count	= cfg.sampleCount;
		desc.SampleDesc.Quality	= 0;
	}

	//------------------------------------------------------------------------

	void
	D3D10Texture2D::SupplementConfig( TexConfigType& cfg, const DescType& desc )
	{
		cfg.width	= desc.Width;
		cfg.height	= desc.Height;
	}

	//------------------------------------------------------------------------

	void
	D3D10Texture2D::FillSRViewDesc( D3D10_SHADER_RESOURCE_VIEW_DESC& srv ) const
	{
		if( GetConfig().cubeTexture )
			srv.ViewDimension = D3D10_SRV_DIMENSION_TEXTURECUBE;
		else
			srv.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
	}

	//------------------------------------------------------------------------
	/*static*/

	void
	D3D10Texture2D::FillInitialData( const TexConfigType::SubresCfg& srcfg, D3D10SubresData::value_type& oSrDataElem )
	{
		oSrDataElem.SysMemPitch	= (UINT)srcfg.pitch;
	}

	//------------------------------------------------------------------------
	/*static*/

	UINT32
	D3D10Texture2D::GetSubresCount( const TexConfigType& cfg )
	{
		if( cfg.cubeTexture )
			return 6 * cfg.numMips;
		else
			return Parent::GetSubresCount( cfg );
	}

}

