#ifndef PROVIDERS_EFFECTPROVIDERCOMMON_H_INCLUDED
#define PROVIDERS_EFFECTPROVIDERCOMMON_H_INCLUDED

#include "Forw.h"

#include "EffectKey.h"

namespace Mod
{

	struct EffectCompiler 
	{
		virtual bool Compile( const Bytes& shlangCode, const EffectKey::Defines& defines, Bytes& oCode, String& oErrors ) const = 0;
	};

	void	FillEffectConfigBaseWithCompiledCode( const EffectCompiler& cpler, EffectProviderConfigBase pcfg, const EffectKey& effkey, EffectConfigBase& oCfg );
}

#endif