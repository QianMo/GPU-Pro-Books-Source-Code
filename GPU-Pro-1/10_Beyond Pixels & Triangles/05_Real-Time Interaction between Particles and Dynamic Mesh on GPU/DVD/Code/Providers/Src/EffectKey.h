#ifndef PROVIDERS_EFFECTKEY_H_INCLUDED
#define PROVIDERS_EFFECTKEY_H_INCLUDED

#include "Forw.h"

#include "ExportDefs.h"

namespace Mod
{
	struct EffectKey
	{
		typedef Types< EffectDefine > :: Set Defines;

		String		file;
		String		poolFile;
		Defines		defines;

		EXP_IMP EffectKey( );
		// NOTE : intentionally implicit
		EXP_IMP EffectKey( const String& a_file );
		EXP_IMP EffectKey( const WCHAR* a_file );
	};
}

#endif