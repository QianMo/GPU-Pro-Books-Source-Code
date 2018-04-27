#ifndef PROVIDERS_EFFECTSUBVARIATIONMAP_H_INCLUDED
#define PROVIDERS_EFFECTSUBVARIATIONMAP_H_INCLUDED

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EffectSubVariationMapNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class EffectSubVariationMap : public EffectSubVariationMapNS::ConfigurableImpl<EffectSubVariationMapConfig>
	{
		// types
	public:

		// constructors / destructors
	protected:
		EXP_IMP explicit EffectSubVariationMap( const EffectSubVariationMapConfig& cfg );
		EXP_IMP ~EffectSubVariationMap();
	
		// manipulation/ access
	public:
		EXP_IMP EffectDefines			GetEffectDefines( EffectVariationID subId, EffectSubVariationBits bits ) const;
		EXP_IMP EffectSubVariationBits	GetSubVariationBits( const Strings& supportedSubVariations ) const;

		// prolymorphism
	private:
		virtual	void					GetEffectDefinesImpl( EffectDefines& oDefines, EffectVariationID subId, EffectSubVariationBits bits ) const	= 0;
		virtual EffectSubVariationBits	GetSubVariationBitsImpl( const Strings& supportedSubVariations ) const = 0;


	};
}

#endif