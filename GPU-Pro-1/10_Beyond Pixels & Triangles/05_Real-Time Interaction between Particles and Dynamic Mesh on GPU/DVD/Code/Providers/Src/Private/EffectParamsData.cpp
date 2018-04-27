#include "Precompiled.h"

#include "EffectParamsDataConfig.h"
#include "EffectParamsData.h"

#define MD_NAMESPACE EffectParamsDataNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	template class EffectParamsDataNS::ConfigurableImpl<EffectParamsDataConfig>;

	EXP_IMP
	EffectParamsData::EffectParamsData( const EffectParamsDataConfig& cfg ) :
	Base( cfg )
	{
	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectParamsData::~EffectParamsData() 
	{
	}
}