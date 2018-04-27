#ifndef D3D10DRV_D3D10BUFFERCOMMON_H_INCLUDED
#define D3D10DRV_D3D10BUFFERCOMMON_H_INCLUDED

#include "Wrap3D/Src/Forw.h"

namespace Mod
{
	UINT32 AddExtraFlags( const FormattedBufferConfig& cfg, UINT32 flags );
	UINT32 AddExtraFlags( const BufferConfig& cfg, UINT32 flags );
}

#endif