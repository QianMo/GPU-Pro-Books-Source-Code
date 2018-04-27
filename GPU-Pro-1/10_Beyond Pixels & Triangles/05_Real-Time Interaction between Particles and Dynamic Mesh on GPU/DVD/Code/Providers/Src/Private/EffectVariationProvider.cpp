#include "Precompiled.h"

#include "Common/Src/XMLElemConfig.h"
#include "Common/Src/XMLDoc.h"
#include "Common/Src/XMLDocConfig.h"
#include "Common/Src/XIElemArray.h"

#include "EffectVariationConfig.h"
#include "EffectVariation.h"

#include "EffectVariationProviderConfig.h"
#include "EffectVariationProvider.h"

#define MD_NAMESPACE EffectVariationProviderNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{

	//------------------------------------------------------------------------

	template class EffectVariationProviderNS::ConfigurableImpl<EffectVariationProviderConfig>;

	//------------------------------------------------------------------------

	EXP_IMP
	EffectVariationProvider::EffectVariationProvider( const EffectVariationProviderConfig& cfg ) : 
	Parent( cfg )
	{
		struct Convert
		{
			EffectVariationPtr operator () ( const XMLElemPtr& el ) const
			{
				EffectVariationConfig cfg;
				cfg.xmlElem = el;			
				return EffectVariationPtr( new EffectVariation( cfg ) );
			}
		} convert;

		AddItemsFromXMLDoc( cfg.docBytes, convert );

		if( !GetExistingItem(L"Default") )
		{
			EffectVariationConfig cfg;
			XMLElemConfig ecfg;
			ecfg.name =	L"Variation";
			cfg.xmlElem.reset( new XMLElem( ecfg ) );
			AddItem( L"Default", EffectVariationPtr ( new EffectVariation( cfg ) ) );
		}
	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectVariationProvider::~EffectVariationProvider()
	{

	}


}