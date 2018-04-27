#ifndef PROVIDERS_MODELCMPTDEFPROVIDER_H_INCLUDED
#define PROVIDERS_MODELCMPTDEFPROVIDER_H_INCLUDED

#include "Forw.h"

#include "Provider.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ModelCmptDefProviderNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class ModelCmptDefProvider :	public Provider		
														<
												DefaultProvTConfig	<
																		ModelCmptDefProvider,
																		ModelCmptDef,
																		ModelCmptDefProviderNS::ConfigurableImpl<ModelCmptDefProviderConfig>
																	>
														>
	{
		friend Parent;
		// construction/ destruction
	public:
		EXP_IMP explicit ModelCmptDefProvider( const ModelCmptDefProviderConfig& cfg );
		EXP_IMP ~ModelCmptDefProvider();

		// manipulation/ access
	public:
		String GetFilePath( const KeyType& key ) const;

		// polymorphism(static)
	private:

		ItemTypePtr CreateItemImpl( const KeyType& key );

	};
}

#endif
