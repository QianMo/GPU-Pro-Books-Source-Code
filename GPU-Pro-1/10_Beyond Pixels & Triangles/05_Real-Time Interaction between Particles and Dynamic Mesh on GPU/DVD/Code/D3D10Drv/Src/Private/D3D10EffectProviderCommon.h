#ifndef D3D10DRV_D3D10EFFECTPROVIDERCOMMON_H_INCLUDED
#define D3D10DRV_D3D10EFFECTPROVIDERCOMMON_H_INCLUDED

#include "Providers/Src/Forw.h"
#include "Providers/Src/EffectKey.h"

#include "Forw.h"

namespace Mod
{
	bool D3D10CompileEffectImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, UINT effectCompileFlags, Bytes& oCode, String& oErrors );
}

#endif