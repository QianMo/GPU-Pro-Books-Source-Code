#ifndef	PROVIDERS_EFFECTDEFPROVIDER_H_INCLUDED
#define PROVIDERS_EFFECTDEFPROVIDER_H_INCLUDED

#include "Forw.h"

#include "Provider.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EffectDefProviderNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class EffectDefProvider : public Provider	<
										DefaultProvTConfig	< 
																	EffectDefProvider,
																	EffectDef,
																	EffectDefProviderNS::ConfigurableImpl<EffectDefProviderConfig>
															>
												>
	{
		// construction/ destruction
	public:
		EXP_IMP explicit EffectDefProvider( const EffectDefProviderConfig& cfg );
		EXP_IMP ~EffectDefProvider();

		// helpers
	private:
		void CheckConsistensy();

	};
}

#endif