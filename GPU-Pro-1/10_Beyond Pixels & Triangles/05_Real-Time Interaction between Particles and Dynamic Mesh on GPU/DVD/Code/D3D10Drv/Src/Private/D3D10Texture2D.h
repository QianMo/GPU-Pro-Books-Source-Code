#ifndef D3D10_D3D10TEXTURE2D_H_INCLUDED
#define D3D10_D3D10TEXTURE2D_H_INCLUDED

#include "Wrap3D/Src/TextureConfig.h"

#include "D3D10TextureImpl.h"

namespace Mod
{

	struct Texture2DConfigEx : Texture2DConfig
	{
		explicit Texture2DConfigEx( const Texture2DConfig& cfg );
		explicit Texture2DConfigEx( const TextureCUBEConfig& cfg );

		virtual Texture2DConfigEx* Clone() const OVERRIDE;

		bool cubeTexture;
	};

	struct D3D10Texture2DConfig
	{
		typedef class D3D10Texture2D	Child;
		typedef D3D10_TEXTURE2D_DESC	DescType;
		typedef ID3D10Texture2D			ResType;
		typedef Texture2DConfigEx		TexConfigType;
		static HRESULT ( MD_D3D_CALLING_CONV ID3D10Device::*CreateTexture)( const DescType*, const D3D10_SUBRESOURCE_DATA*, ResType** );
	};

	class D3D10Texture2D : public D3D10TextureImpl<D3D10Texture2DConfig>
	{
		friend Parent;
		friend MD_D3D10_CREATE_TEXTURE_FROM_RESOURCE_FUNCTION;

		// construction/ destruction
	public:
		D3D10Texture2D( const TexConfigType& cfg, ID3D10Device* dev );
		~D3D10Texture2D();

	private:
		D3D10Texture2D( const TexConfigType& cfg, ResourcePtr ptr );

	private:		
		static void		Configure( Parent::DescType& desc, const TexConfigType& cfg );
		static void		SupplementConfig( TexConfigType& cfg, const DescType& desc );
		void			FillSRViewDesc( D3D10_SHADER_RESOURCE_VIEW_DESC& srv ) const;
		static void		FillInitialData( const TexConfigType::SubresCfg& srcfg, D3D10SubresData::value_type& oSrDataElem );
		static UINT32	GetSubresCount( const TexConfigType& cfg );

	};
}


#endif