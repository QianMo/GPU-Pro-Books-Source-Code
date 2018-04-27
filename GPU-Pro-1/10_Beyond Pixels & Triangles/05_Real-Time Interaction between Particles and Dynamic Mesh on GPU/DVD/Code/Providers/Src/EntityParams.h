#ifndef PROVIDERS_ENTITYPARAMS_H_INCLUDED
#define PROVIDERS_ENTITYPARAMS_H_INCLUDED

#include "Wrap3D/Src/Forw.h"

#include "Forw.h"

namespace Mod
{
	struct EntityParams
	{
		BufferPtr		transformBuffer;
		BufferTypesSet	requiredBuffers;
	};
}

#endif