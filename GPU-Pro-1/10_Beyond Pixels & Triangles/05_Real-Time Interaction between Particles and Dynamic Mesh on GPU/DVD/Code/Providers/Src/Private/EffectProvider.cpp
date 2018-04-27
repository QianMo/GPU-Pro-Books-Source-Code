#include "Precompiled.h"

#include "Wrap3D/Src/EffectConfig.h"
#include "Wrap3D/Src/Device.h"

#include "EffectPoolProvider.h"
#include "Providers.h"

#include "EffectProviderCommon.h"

#include "EffectProviderConfig.h"
#include "EffectProvider.h"

#define MD_NAMESPACE EffectProviderNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{

	template class EffectProviderNS::ConfigurableImpl<EffectProviderConfig>;

	//------------------------------------------------------------------------

	EXP_IMP
	EffectProvider::EffectProvider( const EffectProviderConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectProvider::~EffectProvider()
	{

	}

	//------------------------------------------------------------------------

	EffectPtr
	EffectProvider::CreateItemImpl( const EffectKey& key )
	{
		const ConfigType& cfg = GetConfig();

		EffectConfig ecfg;

		struct EffectCompilerImpl : EffectCompiler
		{
			EffectCompilerImpl( EffectProvider* a_prov, bool a_child ) :
			prov( a_prov ),
			child( a_child )
			{

			}

			virtual bool Compile( const Bytes& shlangCode, const EffectKey::Defines& defines, Bytes& oCode, String& oErrors ) const OVERRIDE
			{
				return prov->CompileEffectImpl( shlangCode, defines, child, oCode, oErrors );
			}

			EffectProvider*	prov;
			bool			child;
		} cpler ( this, !key.poolFile.empty() );

		if( !key.poolFile.empty() )
		{
			ecfg.pool = Providers::Single().GetEffectPoolProv()->GetItem( key.poolFile );
		}

		FillEffectConfigBaseWithCompiledCode( cpler, cfg, key, ecfg );
			
		return cfg.dev->CreateEffect( ecfg );
	}

	//------------------------------------------------------------------------

	bool operator < ( const EffectKey& key1, const EffectKey& key2 )
	{
		if( key1.file < key2.file )
			return true;
		else
		if( key2.file < key1.file )
			return false;
		else
		{
			if( key1.poolFile < key2.poolFile )
				return true;
			else
			if( key2.poolFile < key1.poolFile )
				return false;
			else
			{
				size_t	s1 = key1.defines.size(), 
						s2 = key2.defines.size();
				if( s1 < s2 )
					return true;
				else
				if( s2 < s1 )
					return false;
				else
				{
					for( EffectKey::Defines::const_iterator		i1 = key1.defines.begin(),
																i2 = key2.defines.begin(),
																e1 = key1.defines.end();
																i1 != e1;
																++i1, ++i2
																 )
					{
						if( *i1 < *i2 )
							return true;
						if( *i2 < *i1 )
							return false;
					}
				}
			}
		}

		return false;
	}

}