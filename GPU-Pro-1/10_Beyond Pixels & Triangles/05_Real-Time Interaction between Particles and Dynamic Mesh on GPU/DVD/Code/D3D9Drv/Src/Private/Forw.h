#ifndef D3D9DRV_FORW_H_INCLUDED
#define D3D9DRV_FORW_H_INCLUDED

namespace Mod
{
#include "DefDeclareClass.h"

	MOD_DECLARE_CLASS(D3D9FormatMap);
	MOD_DECLARE_CLASS(D3D9UsageMap);
	MOD_DECLARE_CLASS(D3D9EffectStateManager)
	MOD_DECLARE_CLASS(D3D9TextureCoordinator)

	class D3D9InputLayout;
	class D3D9Device;

#include "UndefDeclareClass.h"	

	const D3DFORMAT D3D9_BACKBUFFER_FORMAT	= D3DFMT_A8R8G8B8;
	const D3DFORMAT D3D9_DISPLAY_FORMAT		= D3DFMT_X8R8G8B8;	

	struct D3D9CapsConstants;

}

#endif