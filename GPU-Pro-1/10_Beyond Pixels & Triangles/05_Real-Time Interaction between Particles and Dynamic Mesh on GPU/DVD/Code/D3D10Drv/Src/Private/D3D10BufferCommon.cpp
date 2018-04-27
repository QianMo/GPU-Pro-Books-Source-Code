#include "Precompiled.h"
#include "Wrap3D/Src/BufferConfig.h"

namespace Mod
{
	UINT32
	AddExtraFlags( const FormattedBufferConfig& cfg, UINT32 flags )
	{
		return cfg.renderTarget ? D3D10_BIND_RENDER_TARGET | flags : flags;
	}

	//------------------------------------------------------------------------

	UINT32
	AddExtraFlags( const BufferConfig& cfg, UINT32 flags )
	{
		return cfg, flags;
	}
}