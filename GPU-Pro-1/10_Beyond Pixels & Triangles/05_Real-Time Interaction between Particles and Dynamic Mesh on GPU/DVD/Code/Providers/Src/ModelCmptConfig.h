#ifndef PROVIDERS_MODELCMPTCONFIG_H_INCLUDED
#define PROVIDERS_MODELCMPTCONFIG_H_INCLUDED

#include "Math/Src/Forw.h"

#include "Wrap3D/Src/Forw.h"

#include "Forw.h"

#include "ExportDefs.h"

#ifdef _MSC_VER
#pragma warning( disable : 4505 )
#endif

namespace Mod
{
	struct ModelCmptConfig
	{
		DevicePtr					dev;
		ModelCmptDefPtr				def;
		ModelComponentFlags			flags;
		String						baseKey;

		EXP_IMP ModelCmptConfig();
		EXP_IMP virtual ~ModelCmptConfig();
		virtual ModelCmptConfig* Clone() const = 0;
	};

	//------------------------------------------------------------------------

	struct VertexIndexModelCmptConfigBase : ModelCmptConfig
	{
		Math::BBoxPtr bbox;
	};


	//------------------------------------------------------------------------

	struct VertexIndexModelCmptConfig : VertexIndexModelCmptConfigBase
	{
		EXP_IMP virtual VertexIndexModelCmptConfig* Clone() const OVERRIDE;
	};
}

#endif
