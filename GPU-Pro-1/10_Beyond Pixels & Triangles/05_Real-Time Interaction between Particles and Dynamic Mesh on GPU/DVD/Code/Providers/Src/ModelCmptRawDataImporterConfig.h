#ifndef PROVIDERS_MODELCMPTRAWDATAIMPORTERCONFIG_H_INCLUDED
#define PROVIDERS_MODELCMPTRAWDATAIMPORTERCONFIG_H_INCLUDED

#include "ExportDefs.h"

namespace Mod
{
	struct ModelCmptRawDataImporterConfigBase
	{
		virtual ~ModelCmptRawDataImporterConfigBase();

		virtual ModelCmptRawDataImporterConfigBase* Clone() const = 0;
	};


	//------------------------------------------------------------------------

	struct ModelCmptRawDataImporterConfig : ModelCmptRawDataImporterConfigBase
	{
		virtual ModelCmptRawDataImporterConfig* Clone() const OVERRIDE;
	};


}

#endif