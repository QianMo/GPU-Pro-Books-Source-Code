#include "Precompiled.h"

#include "Wrap3D/Src/Exports.h"

#include "Common/Src/StringUtils.h"

#include "D3D9Instance.h"

namespace Mod
{

	struct D3D9GetShutDownSetter
	{
		D3D9GetShutDownSetter()
		{
			W3DSetShutDownFunction( ShutDown );
		}

		static void ShutDown()
		{
			if( D3D9Instance::Exists() )
			{
				D3D9Instance::Single().Release();
			}
		}
	} gD3D10GetShutDownSetter;

}