#include "Precompiled.h"

#include "EffectDefine.h"

#include "EffectSubVariationMapConfig.h"
#include "EffectSubVariationMap.h"

#define MD_NAMESPACE EffectSubVariationMapNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	EffectSubVariationMap::EffectSubVariationMap( const EffectSubVariationMapConfig& cfg ) :
	Parent( cfg )
	{
	}

	//------------------------------------------------------------------------

	EffectSubVariationMap::~EffectSubVariationMap() 
	{
	}

	//------------------------------------------------------------------------

	EffectDefines
	EffectSubVariationMap::GetEffectDefines( EffectVariationID subId, EffectSubVariationBits bits ) const
	{	
		EffectDefines result;
		GetEffectDefinesImpl( result, subId, bits );
		return result;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectSubVariationBits
	EffectSubVariationMap::GetSubVariationBits( const Strings& supportedVariations ) const
	{
		return GetSubVariationBitsImpl( supportedVariations );
	}



}