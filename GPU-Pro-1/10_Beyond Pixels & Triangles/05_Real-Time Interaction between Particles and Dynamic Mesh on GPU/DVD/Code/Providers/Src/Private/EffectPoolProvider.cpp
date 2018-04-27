#include "Precompiled.h"

#include "Wrap3D/Src/EffectPoolConfig.h"
#include "Wrap3D/Src/Device.h"

#include "EffectDefine.h"

#include "EffectProviderCommon.h"

#include "EffectPoolProviderConfig.h"
#include "EffectPoolProvider.h"

#define MD_NAMESPACE EffectPoolProviderNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{

	template class EffectPoolProviderNS::ConfigurableImpl<EffectPoolProviderConfig>;

	EXP_IMP
	EffectPoolProvider::EffectPoolProvider( const EffectPoolProviderConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectPoolProvider::~EffectPoolProvider()
	{

	}

	//------------------------------------------------------------------------

	EffectPoolPtr
	EffectPoolProvider::CreateItemImpl( const String& key )
	{
		const ConfigType& cfg = GetConfig();

		EffectPoolConfig ecfg;

		struct EffectCompilerImpl : EffectCompiler
		{
			EffectCompilerImpl( EffectPoolProvider* a_prov ) :
			prov( a_prov )
			{

			}

			virtual bool Compile( const Bytes& shlangCode, const EffectKey::Defines& defines, Bytes& oCode, String& oErrors ) const OVERRIDE
			{
				return prov->CompileEffectPoolImpl( shlangCode, defines, oCode, oErrors );
			}

			EffectPoolProvider* prov;
		} cpler ( this );

		FillEffectConfigBaseWithCompiledCode( cpler, cfg, key, ecfg );

		return cfg.dev->CreateEffectPool( ecfg );

	}

}
