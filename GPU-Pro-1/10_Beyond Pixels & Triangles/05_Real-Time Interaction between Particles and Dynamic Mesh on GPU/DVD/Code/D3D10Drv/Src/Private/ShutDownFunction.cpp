#include "Precompiled.h"

#include "Wrap3D/Src/Exports.h"

#include "Common/Src/StringUtils.h"

#include "DXGIFactory.h"

namespace Mod
{

	struct D3D10GetShutDownSetter
	{
		D3D10GetShutDownSetter()
		{
			W3DSetShutDownFunction( ShutDown );
		}

		static void ShutDown()
		{
			DXGIFactory::Single().Release();
		}
	} gD3D10GetShutDownSetter;

}