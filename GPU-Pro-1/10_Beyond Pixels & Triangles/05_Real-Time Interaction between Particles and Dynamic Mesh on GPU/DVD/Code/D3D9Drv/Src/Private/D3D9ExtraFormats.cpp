#include "Precompiled.h"

#include "D3D9ExtraFormats.h"

namespace Mod
{
	bool IsFormatExtra( UINT32 fmt )
	{
		switch ( fmt )
		{
		case MDFMT_R32G32B32_FLOAT:
			return true;
		default:
			return false;
		}
	}
}