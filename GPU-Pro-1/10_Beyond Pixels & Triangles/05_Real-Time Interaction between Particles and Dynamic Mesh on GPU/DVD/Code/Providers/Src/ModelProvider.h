#ifndef PROVIDERS_MODELPROVIDER_H_INCLUDED
#define PROVIDERS_MODELPROVIDER_H_INCLUDED

#include "Forw.h"

#include "Provider.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ModelProviderNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class ModelProvider :	public Provider		<
										DefaultProvTConfig	<
																ModelProvider,
																Model,
																ModelProviderNS::ConfigurableImpl<ModelProviderConfig>
															>
												>
	{
		friend Parent;

		// construction/ destruction
	public:
		EXP_IMP explicit ModelProvider( const ModelProviderConfig& cfg );
		EXP_IMP ~ModelProvider();

	public:
		using Parent::RemoveItem;

		// static polymorphism
	private:
		ItemTypePtr CreateItemImpl( const KeyType& key );
	};
}

#endif
