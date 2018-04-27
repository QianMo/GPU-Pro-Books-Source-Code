#include "Precompiled.h"

#include "EffectDefine.h"

namespace Mod
{
	EXP_IMP bool operator < ( const EffectDefine& def1, const EffectDefine& def2 )
	{
		if( def1.name < def2.name )
			return true;
		else
		if( def2.name < def1.name )
			return false;
		else
			return def1.val < def2.val;
	}
}