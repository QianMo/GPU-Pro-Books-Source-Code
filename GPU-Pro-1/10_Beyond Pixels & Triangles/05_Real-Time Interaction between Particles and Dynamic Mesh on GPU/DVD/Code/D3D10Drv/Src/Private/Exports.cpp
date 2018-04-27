#include "Precompiled.h"
#include "PlatformDll.h"
#include "D3D10Device.h"
#include "D3D10EffectProvider.h"
#include "D3D10EffectPoolProvider.h"

#include "Wrap3D\Src\Forw.h"

namespace Mod
{

#ifdef MD_WIN64_PLATFORM

#pragma comment(linker, "/EXPORT:CreateDevice=?CreateDevice@Mod@@YA?AV?$shared_ptr@VDevice@Mod@@@boost@@AEBUDeviceConfig@1@@Z")
#pragma comment(linker, "/EXPORT:CreateEffectProvider=?CreateEffectProvider@Mod@@YA?AV?$shared_ptr@VEffectProvider@Mod@@@boost@@AEBUEffectProviderConfig@1@@Z")
#pragma comment(linker, "/EXPORT:CreateEffectPoolProvider=?CreateEffectPoolProvider@Mod@@YA?AV?$shared_ptr@VEffectPoolProvider@Mod@@@boost@@AEBUEffectPoolProviderConfig@1@@Z")

#elif defined(MD_WIN32_PLATFORM)

#pragma comment(linker, "/EXPORT:CreateDevice=?CreateDevice@Mod@@YA?AV?$shared_ptr@VDevice@Mod@@@boost@@ABUDeviceConfig@1@@Z") 
#pragma comment(linker, "/EXPORT:CreateEffectProvider=?CreateEffectProvider@Mod@@YA?AV?$shared_ptr@VEffectProvider@Mod@@@boost@@ABUEffectProviderConfig@1@@Z")
#pragma comment(linker, "/EXPORT:CreateEffectPoolProvider=?CreateEffectPoolProvider@Mod@@YA?AV?$shared_ptr@VEffectPoolProvider@Mod@@@boost@@ABUEffectPoolProviderConfig@1@@Z")

#else
#error Unsupported platform
#endif

	DLLEXPORT DevicePtr CreateDevice( const DeviceConfig& cfg )
	{
		return DevicePtr( new D3D10Device( cfg ) );
	}

	//------------------------------------------------------------------------

	DLLEXPORT EffectProviderPtr	CreateEffectProvider( const EffectProviderConfig& cfg )
	{
		return EffectProviderPtr( new D3D10EffectProvider( cfg ) );
	}

	//------------------------------------------------------------------------

	DLLEXPORT EffectPoolProviderPtr	CreateEffectPoolProvider( const EffectPoolProviderConfig& cfg )
	{
		return EffectPoolProviderPtr( new D3D10EffectPoolProvider( cfg ) );
	}

}