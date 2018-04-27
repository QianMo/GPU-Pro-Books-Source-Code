#ifndef PROVIDERS_MODELCMPTRAWDATAPROVIDER_H_INCLUDED
#define PROVIDERS_MODELCMPTRAWDATAPROVIDER_H_INCLUDED

#include "Forw.h"

#include "Provider.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ModelCmptRawDataProviderNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class ModelCmptRawDataProvider : public Provider	<
											DefaultProvTConfig	< 
																	ModelCmptRawDataProvider,
																	ModelCmptRawDataArray,
																	ModelCmptRawDataProviderNS::ConfigurableImpl<ModelCmptRawDataProviderConfig>
																>
														>
	{
		friend Parent;

		// construction/ destruction
	public:
		EXP_IMP explicit ModelCmptRawDataProvider( const ModelCmptRawDataProviderConfig& cfg );
		EXP_IMP ~ModelCmptRawDataProvider();

		// manipulation/ access
	public:
		String GetFilePath( const KeyType& key ) const;

		// static polymorphism
	private:
		ItemTypePtr CreateItemImpl( const KeyType& key );
	};
}

#endif
