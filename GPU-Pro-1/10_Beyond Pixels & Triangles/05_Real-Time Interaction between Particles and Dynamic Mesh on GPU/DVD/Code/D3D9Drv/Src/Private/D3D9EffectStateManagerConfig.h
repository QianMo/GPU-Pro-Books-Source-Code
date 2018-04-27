#ifndef D3D9DRV_D3D9EFFECTSTATEMANAGERCONFIG_H_INCLUDED
#define D3D9DRV_D3D9EFFECTSTATEMANAGERCONFIG_H_INCLUDED

#include "Wrap3D/Src/Forw.h"
#include "Forw.h"

namespace Mod
{
	struct D3D9EffectStateManagerConfig
	{
		ComPtr < IDirect3DDevice9 >		device;
		D3D9TextureCoordinatorPtr		texCoordinator;
	};
}

#endif
