#ifndef D3D9DRV_D3D9TEXTURECOORDINATORCONFIG_H_INCLUDED
#define D3D9DRV_D3D9TEXTURECOORDINATORCONFIG_H_INCLUDED

#include "Forw.h"

namespace Mod
{
	struct D3D9TextureCoordinatorConfig
	{
		ComPtr< IDirect3DDevice9 >	device;
		UINT32						numTextures;
	};
}

#endif