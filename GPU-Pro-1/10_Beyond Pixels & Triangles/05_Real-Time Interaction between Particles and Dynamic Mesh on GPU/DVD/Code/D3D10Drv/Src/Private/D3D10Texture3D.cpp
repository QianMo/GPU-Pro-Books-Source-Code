#include "Precompiled.h"

#include "Wrap3D\Src\TextureConfig.h"

#include "D3D10Texture3D.h"

namespace Mod
{

	HRESULT ( MD_D3D_CALLING_CONV ID3D10Device::*D3D10Texture3DConfig::CreateTexture)( const DescType*, const D3D10_SUBRESOURCE_DATA*, ResType** ) = &ID3D10Device::CreateTexture3D;

	//------------------------------------------------------------------------

	D3D10Texture3D::D3D10Texture3D( const TexConfigType& cfg, ID3D10Device* dev ) :
	Parent( cfg, dev )
	{

	}

	//------------------------------------------------------------------------

	D3D10Texture3D::D3D10Texture3D( const TexConfigType& cfg, ResourcePtr res ) : 
	Parent( cfg, res )
	{

	}

	//------------------------------------------------------------------------

	D3D10Texture3D::~D3D10Texture3D( )
	{

	}

	//------------------------------------------------------------------------

	void
	D3D10Texture3D::Configure( Parent::DescType& desc, const TexConfigType& cfg )
	{
		MD_FERROR_ON_FALSE( cfg.arraySize == 1 );

		desc.Width	= cfg.width;
		desc.Height	= cfg.height;
		desc.Depth	= cfg.depth;
	}

	//------------------------------------------------------------------------

	void
	D3D10Texture3D::SupplementConfig( TexConfigType& cfg, const DescType& desc )
	{
		cfg.width	= desc.Width;
		cfg.height	= desc.Height;
		cfg.depth	= desc.Depth;
	}

	//------------------------------------------------------------------------

	void
	D3D10Texture3D::FillSRViewDesc( D3D10_SHADER_RESOURCE_VIEW_DESC& srv ) const
	{
		srv.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE3D;
	}

	//------------------------------------------------------------------------
	/*static*/
	
	void
	D3D10Texture3D::FillInitialData( const TexConfigType::SubresCfg& srcfg, D3D10SubresData::value_type& oSrDataElem )
	{
		oSrDataElem.SysMemPitch			= (UINT)srcfg.rowPitch;
		oSrDataElem.SysMemSlicePitch	= (UINT)srcfg.slicePitch;
	}

}