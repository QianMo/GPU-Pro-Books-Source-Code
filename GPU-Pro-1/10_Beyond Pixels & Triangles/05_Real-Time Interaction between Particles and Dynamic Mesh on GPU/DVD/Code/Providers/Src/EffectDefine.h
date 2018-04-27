#ifndef PROVIDERS_EFFECTDEFINE_H_INCLUDED
#define PROVIDERS_EFFECTDEFINE_H_INCLUDED

#include "ExportDefs.h"

namespace Mod
{
	struct EffectDefine
	{
		AnsiString name;
		AnsiString val;
	};

	EXP_IMP bool operator < ( const EffectDefine& def1, const EffectDefine& def2 );
}


#endif