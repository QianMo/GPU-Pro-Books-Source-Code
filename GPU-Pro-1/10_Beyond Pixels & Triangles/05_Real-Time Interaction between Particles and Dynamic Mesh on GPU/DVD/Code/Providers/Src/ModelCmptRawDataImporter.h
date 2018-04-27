#ifndef PROVIDERS_MODELCMPTRAWDATAIMPORTER_H_INCLUDED
#define PROVIDERS_MODELCMPTRAWDATAIMPORTER_H_INCLUDED

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ModelCmptRawDataImporterNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class ModelCmptRawDataImporter : public ModelCmptRawDataImporterNS::ConfigurableImpl< ModelCmptRawDataImporterConfig >
	{
		// construction/ destruction
	public:
		explicit ModelCmptRawDataImporter( const ModelCmptRawDataImporterConfig& cfg );
		virtual ~ModelCmptRawDataImporter();

		// manipulation/ access
	public:
		void Import( const String& fileName, ModelCmptRawDataArray& oResult );

		// polymorphism
	private:		
		virtual void ImportImpl( const String& fileName, ModelCmptRawDataArray& oResult ) = 0;
	};
}

#endif