#ifndef ENGINE_EXPORTS_H_INCLUDED
#define ENGINE_EXPORTS_H_INCLUDED

#include "Forw.h"
#include "ExportDefs.h"

namespace Mod
{
	EXP_IMP bool CreateEngine( const EngineConfig& cfg, void (*UserInit)() = NULL );
	EXP_IMP void DestroyEngine();
}

#endif