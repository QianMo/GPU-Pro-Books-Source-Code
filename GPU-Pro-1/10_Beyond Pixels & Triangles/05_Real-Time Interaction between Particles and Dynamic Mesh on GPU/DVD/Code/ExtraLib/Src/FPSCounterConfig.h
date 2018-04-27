#ifndef EXTRALIB_FPSCOUNTERCONFIG_H_INCLUDED
#define EXTRALIB_FPSCOUNTERCONFIG_H_INCLUDED

#include "WrapSys/Src/Forw.h"

#include "Forw.h"

namespace Mod
{

	struct FPSCounterConfig
	{
		FPSCounterConfig();

		float		avarageSpan;
		TimerPtr	timer;
	};

	//------------------------------------------------------------------------

	inline
	FPSCounterConfig::FPSCounterConfig() :
	avarageSpan( 0.25f )
	{

	}
}

#endif