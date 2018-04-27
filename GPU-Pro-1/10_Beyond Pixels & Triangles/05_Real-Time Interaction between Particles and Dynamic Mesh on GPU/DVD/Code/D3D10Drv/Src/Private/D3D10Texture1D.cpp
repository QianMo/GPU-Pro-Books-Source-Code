#include "Precompiled.h"

#include "Wrap3D\Src\TextureConfig.h"

#include "D3D10Format.h"
#include "D3D10Usage.h"
#include "D3D10Exception.h"

#include "D3D10Texture1D.h"

namespace Mod
{

	//------------------------------------------------------------------------

	HRESULT ( MD_D3D_CALLING_CONV ID3D10Device::*D3D10Texture1DConfig::CreateTexture)( const DescType*, const D3D10_SUBRESOURCE_DATA*, ResType** ) = &ID3D10Device::CreateTexture1D;

	//------------------------------------------------------------------------

	D3D10Texture1D::D3D10Texture1D( const TexConfigType& cfg, ID3D10Device* dev ) :
	Parent( cfg, dev )
	{
	}

	//------------------------------------------------------------------------

	D3D10Texture1D::D3D10Texture1D( const Texture1DConfig& cfg, ResourcePtr res ):
	Parent( cfg, res )
	{

	}

	//------------------------------------------------------------------------

	D3D10Texture1D::~D3D10Texture1D()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D10Texture1D::Configure( Parent::DescType& desc, const TexConfigType& cfg )
	{
		desc.ArraySize	= cfg.arraySize;
		desc.Width		= cfg.length;
	}

	//------------------------------------------------------------------------

	void
	D3D10Texture1D::SupplementConfig( TexConfigType& cfg, const DescType& desc )
	{
		cfg.length = desc.Width;
	}

	//------------------------------------------------------------------------

	void
	D3D10Texture1D::FillSRViewDesc( D3D10_SHADER_RESOURCE_VIEW_DESC& srv ) const
	{
		srv.ViewDimension	= D3D10_SRV_DIMENSION_TEXTURE1D;		
	}

	//------------------------------------------------------------------------
	/*static*/

	void
	D3D10Texture1D::FillInitialData( const TexConfigType::SubresCfg& srcfg, D3D10SubresData::value_type& oSrDataElem )
	{
		MD_SUPPRESS_UNUSED(oSrDataElem)
		MD_SUPPRESS_UNUSED(srcfg)
	}

}