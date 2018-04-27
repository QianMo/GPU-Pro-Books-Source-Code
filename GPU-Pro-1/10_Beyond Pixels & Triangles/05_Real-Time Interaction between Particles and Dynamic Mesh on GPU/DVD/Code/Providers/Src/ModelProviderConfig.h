#ifndef PROVIDERS_MODELPROVIDERCONFIG_H_INCLUDED
#define PROVIDERS_MODELPROVIDERCONFIG_H_INCLUDED

#include "Forw.h"

namespace Mod
{
	struct ModelProviderConfig
	{
		String						path;
		VFontDefProviderPtr			fontDefProvider;
		DevicePtr					dev;
		bool						forceCache;
	};
}

#endif