#ifndef PROVIDERS_EFFECTPOOLPROVIDER_H_INCLUDED
#define PROVIDERS_EFFECTPOOLPROVIDER_H_INCLUDED

#include "Wrap3D/Src/Forw.h"

#include "Forw.h"

#include "Provider.h"

#include "EffectKey.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EffectPoolProviderNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class EffectPoolProvider :	public Provider		<
										DefaultProvTConfig	<
																EffectPoolProvider,
																EffectPool,
																EffectPoolProviderNS::ConfigurableImpl<EffectPoolProviderConfig>
															>
												>
	{
		friend Parent;

		// construction/ destruction
	public:
		EXP_IMP explicit EffectPoolProvider( const EffectPoolProviderConfig& cfg );
		EXP_IMP ~EffectPoolProvider();

		// polymorphism
	private:
		EffectPoolPtr CreateItemImpl( const String& key );

		// polymorphism
	private:
		EXP_IMP virtual bool CompileEffectPoolImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, Bytes& oCode, String& oErrors ) = 0;


	};
}

#endif
