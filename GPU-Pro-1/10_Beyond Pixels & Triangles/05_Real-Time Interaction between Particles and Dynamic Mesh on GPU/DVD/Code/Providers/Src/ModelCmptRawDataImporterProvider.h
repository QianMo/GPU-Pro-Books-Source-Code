#ifndef PROVIDERS_MODELCMPTRAWDATAIMPORTERPROVIDER_H_INCLUDED
#define PROVIDERS_MODELCMPTRAWDATAIMPORTERPROVIDER_H_INCLUDED

#include "Forw.h"

#include "Provider.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ModelCmptRawDataImporterProviderNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class ModelCmptRawDataImporterProvider : public Provider		<
														DefaultProvTConfig	< 
																				ModelCmptRawDataImporterProvider,
																				ModelCmptRawDataImporter,
																				ModelCmptRawDataImporterProviderNS::ConfigurableImpl<ModelCmptRawDataImporterProviderConfig>
																			>
																	>
	{
		// construction/ destruction
	public:
		EXP_IMP explicit ModelCmptRawDataImporterProvider( const ModelCmptRawDataImporterProviderConfig& cfg );
		EXP_IMP ~ModelCmptRawDataImporterProvider();

		// manipulation/ access
	public:
		void RegisterImporter( const String& ext, ModelCmptRawDataImporterPtr importer );		
	};

	void RegisterPrebuiltRawCmptData( ModelCmptRawDataImporterProvider& impProv, const String& key, const ModelCmptRawDataArrayPtr& rawData );
}

#endif
