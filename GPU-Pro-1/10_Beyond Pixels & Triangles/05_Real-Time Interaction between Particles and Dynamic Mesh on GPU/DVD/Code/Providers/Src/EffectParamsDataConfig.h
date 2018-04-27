#ifndef PROVIDERS_EFFECTPARAMSDATACONFIG_H_INCLUDED
#define PROVIDERS_EFFECTPARAMSDATACONFIG_H_INCLUDED

#include "Forw.h"

#include "Common/Src/Forw.h"

namespace Mod
{
	struct EffectParamsDataConfig
	{
		typedef Types< XMLElemPtr > :: Vec XMLElems;

		XMLElems		elems;
		EffectDefPtr	effDef;
	};
}

#endif

