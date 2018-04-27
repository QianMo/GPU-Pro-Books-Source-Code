#ifndef D3D9DRV_EXPORTS_H_INCLUDED
#define D3D9DRV_EXPORTS_H_INCLUDED

#include "PlatformDll.h"
#include "Providers/Src/Forw.h"
#include "Wrap3D/Src/Forw.h"

namespace Mod
{
	DLLIMPORT DevicePtr				CreateDevice( const DeviceConfig& cfg );
	DLLIMPORT EffectProviderPtr		CreateEffectProvider( const EffectProviderConfig& cfg );
	DLLIMPORT EffectPoolProviderPtr	CreateEffectPoolProvider( const EffectPoolProviderConfig& cfg );
}

#endif