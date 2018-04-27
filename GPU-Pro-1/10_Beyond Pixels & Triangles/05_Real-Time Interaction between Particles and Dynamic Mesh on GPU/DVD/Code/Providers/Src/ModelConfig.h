#ifndef PROVIDERS_MODELCONFIG_H_INCLUDED
#define PROVIDERS_MODELCONFIG_H_INCLUDED

#include "Wrap3D/Src/Forw.h"

#include "Forw.h"

#include "ModelCmptEntry.h"

namespace Mod
{
	struct ModelConfig
	{
		typedef Types<ModelCmptEntryPtr> :: Vec CmptEntries;

		CmptEntries entries;
	};
}

#endif