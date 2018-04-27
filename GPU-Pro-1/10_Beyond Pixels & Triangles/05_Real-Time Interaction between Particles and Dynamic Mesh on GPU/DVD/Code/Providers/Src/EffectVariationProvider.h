#ifndef PROVIDERS_EFFECTVARIATIONPROVIDER_H_INCLUDED
#define PROVIDERS_EFFECTVARIATIONPROVIDER_H_INCLUDED

#include "Forw.h"

#include "Provider.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EffectVariationProviderNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class EffectVariationProvider :	public Provider		<
										DefaultProvTConfig	<
																EffectVariationProvider,
																EffectVariation,
																EffectVariationProviderNS::ConfigurableImpl<EffectVariationProviderConfig>
															>
												>
	{
		friend Parent;

		// construction/ destruction
	public:
		EXP_IMP explicit EffectVariationProvider( const EffectVariationProviderConfig& cfg );
		EXP_IMP ~EffectVariationProvider();

	};
}

#endif
