#ifndef PROVIDERS_EFFECTVARIATIONMAP_H_INCLUDED
#define PROVIDERS_EFFECTVARIATIONMAP_H_INCLUDED

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EffectVariationMapNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class EffectVariationMap : public EffectVariationMapNS::ConfigurableImpl<EffectVariationMapConfig>								
	{
		// types & constants
	public:
		typedef Types2< String, EffectVariationID > :: Map	VariationMap;
		typedef Types< EffectVariationPtr > :: Vec			VariationTable;

		// construction/ destruction
	public:
		EXP_IMP explicit EffectVariationMap( const EffectVariationMapConfig& cfg );
		EXP_IMP ~EffectVariationMap();

		// manipulation/ access
	public:
		EXP_IMP EffectVariationID	GetVariationIDByName( const String& name );
		EXP_IMP EffectVariationPtr	GetVariationByID( EffectVariationID id ) const;

		// data
	private:
		VariationMap		mVariationMap;
		VariationTable		mVariationTable;
		EffectVariationID	mLastID;

	};
}

#endif
