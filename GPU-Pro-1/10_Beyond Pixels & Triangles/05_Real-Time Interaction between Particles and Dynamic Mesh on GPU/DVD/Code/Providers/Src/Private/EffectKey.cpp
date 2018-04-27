#include "Precompiled.h"

#include "EffectDefine.h"
#include "EffectKey.h"

namespace Mod
{
	EXP_IMP
	EffectKey::EffectKey( )
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectKey::EffectKey( const String& a_file ) :
	file( a_file )
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectKey::EffectKey( const WCHAR* a_file ) :
	file( a_file )
	{

	}

}