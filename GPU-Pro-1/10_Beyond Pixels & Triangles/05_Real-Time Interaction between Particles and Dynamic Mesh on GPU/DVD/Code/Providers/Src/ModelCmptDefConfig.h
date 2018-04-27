#ifndef PROVIDERS_MODELCMPTDEFCONFIG_H_INCLUDED
#define PROVIDERS_MODELCMPTDEFCONFIG_H_INCLUDED

#include "Common/Src/Forw.h"
#include "Wrap3D/Src/Forw.h"

#include "Forw.h"

namespace Mod
{
	struct ModelCmptDefConfig
	{
		XMLElemPtr				xmlElem;
		DevicePtr				dev;
		String					name;
	};
}

#endif
