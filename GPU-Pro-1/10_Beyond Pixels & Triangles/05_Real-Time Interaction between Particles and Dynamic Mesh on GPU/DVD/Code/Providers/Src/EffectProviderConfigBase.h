#ifndef PROVIDERS_EFFECTPROVIDERCONFIGBASE_H_INCLUDED
#define PROVIDERS_EFFECTPROVIDERCONFIGBASE_H_INCLUDED

#include "Wrap3D/Src/Forw.h"

namespace Mod
{
	struct EffectProviderConfigBase
	{
		bool	autoCreateCache;
		bool	forceCache;

		String path;
		String cachePath;

		String extension;
		String cachedExtension;

		DevicePtr dev;
	};
}

#endif