#ifndef D3D10DRV_TEXTURE1D_H_INCLUDED
#define D3D10DRV_TEXTURE1D_H_INCLUDED

#include "D3D10TextureImpl.h"

namespace Mod
{

	struct D3D10Texture1DConfig
	{
		typedef class D3D10Texture1D	Child;
		typedef D3D10_TEXTURE1D_DESC	DescType;
		typedef ID3D10Texture1D			ResType;
		typedef Texture1DConfig			TexConfigType;
		static HRESULT ( MD_D3D_CALLING_CONV ID3D10Device::*CreateTexture)( const DescType*, const D3D10_SUBRESOURCE_DATA*, ResType** );
	};

	class D3D10Texture1D : public D3D10TextureImpl<D3D10Texture1DConfig>
	{
		friend Parent;
		friend MD_D3D10_CREATE_TEXTURE_FROM_RESOURCE_FUNCTION;
		// construction/ destruction
	public:
		D3D10Texture1D( const Texture1DConfig& cfg, ID3D10Device* dev );
		virtual ~D3D10Texture1D();
	private:
		D3D10Texture1D( const Texture1DConfig& cfg, ResourcePtr res );

	private:		
		static void		Configure( DescType& desc, const TexConfigType& cfg );
		static void		SupplementConfig( TexConfigType& cfg, const DescType& desc );
		void			FillSRViewDesc( D3D10_SHADER_RESOURCE_VIEW_DESC& srv ) const;
		static void		FillInitialData( const TexConfigType::SubresCfg& srcfg, D3D10SubresData::value_type& oSrDataElem );

	};
}

#endif