#include "Precompiled.h"

#include "Wrap3D\Src\TextureConfig.h"

#include "D3D10TextureImpl.h"
#include "D3D10Exception.h"
#include "D3D10Format.h"
#include "D3D10Usage.h"

#include "D3D10Texture1D.h"
#include "D3D10Texture2D.h"
#include "D3D10Texture3D.h"

#include "D3D10FormatMap.h"
#include "D3D10UsageMap.h"

namespace Mod
{

	//------------------------------------------------------------------------

	template <typename Config>
	D3D10TextureImpl<Config>::D3D10TextureImpl( const TextureConfig& cfg, ID3D10Device* dev ) : 
	D3D10Texture( cfg )
	{
		DescType desc = {};

		desc.Format			= static_cast<const D3D10Format*>(cfg.fmt)->GetValue();
		desc.Usage			= static_cast<const D3D10Usage*>(cfg.usage)->GetValue();
		desc.BindFlags		= 0;
		desc.CPUAccessFlags	= 0;
		desc.MiscFlags		= 0;
		desc.MipLevels		= cfg.numMips;

		if( cfg.shaderResource )
			desc.BindFlags |= D3D10_BIND_SHADER_RESOURCE;

		if( cfg.renderTarget )
			desc.BindFlags |= D3D10_BIND_RENDER_TARGET;

		if( cfg.depthStencil )
			desc.BindFlags |= D3D10_BIND_DEPTH_STENCIL;

		// must set array size & texture dimensions (no array size for 3D texture)
		Child::Configure( desc, GetConfig() );

		{
			ResType* tex;

			D3D10SubresData			initData;
			D3D10_SUBRESOURCE_DATA*	pInitData;

			if( cfg.data.GetSize() )
			{
				MD_FERROR_ON_FALSE( cfg.numMips );

				const TexConfigType& tcfg = static_cast<const TexConfigType&>(cfg);

				initData.resize( Child::GetSubresCount( tcfg ) );

				for( UINT32 i = 0, e = (UINT32)initData.size(); i < e; i ++ )
				{
					D3D10_SUBRESOURCE_DATA& srData					= initData.at( i );
					const typename TexConfigType::SubresCfg& srCfg	= tcfg.subresCfgs.at( i );

					srData.pSysMem = &cfg.data[ srCfg.dsp ];
					
					Child::FillInitialData( srCfg, srData );
				}

				pInitData = &initData.at( 0 );
			}
			else
				pInitData = NULL;
			

			D3D10_THROW_IF( ((&*dev)->*Config::CreateTexture)( &desc, pInitData, &tex ) );
			SetResource(tex);
		}

	}

	//------------------------------------------------------------------------

	template <typename Config>
	D3D10TextureImpl<Config>::D3D10TextureImpl( const TextureConfig& cfg, ResourcePtr res ) :
	Base( cfg, res )
	{

	}


	//------------------------------------------------------------------------

	template <typename Config>
	D3D10TextureImpl<Config>::~D3D10TextureImpl()
	{

	}

	//------------------------------------------------------------------------

	template<typename Config>
	const typename D3D10TextureImpl<Config>::TexConfigType&
	D3D10TextureImpl<Config>::GetConfig() const
	{		
		const TextureConfig& cfg = Base::GetConfig();
		MD_CHECK_TYPE(const TexConfigType, &cfg);
		return static_cast<const TexConfigType&>(cfg);
	}

	//------------------------------------------------------------------------

	template<typename Config>
	/*static*/
	UINT32
	D3D10TextureImpl<Config>::GetSubresCount( const TextureConfig& cfg )
	{
		return cfg.arraySize * cfg.numMips;
	}

	//------------------------------------------------------------------------

	// explicitly instantiate
	template D3D10TextureImpl<D3D10Texture1DConfig>;
	template D3D10TextureImpl<D3D10Texture2DConfig>;
	template D3D10TextureImpl<D3D10Texture3DConfig>;

}

