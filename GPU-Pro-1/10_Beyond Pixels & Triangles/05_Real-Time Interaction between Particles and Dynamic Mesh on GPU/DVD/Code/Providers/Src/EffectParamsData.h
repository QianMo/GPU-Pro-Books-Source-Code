#ifndef PROVIDERS_EFFECTPARAMSDATA_H_INCLUDED
#define PROVIDERS_EFFECTPARAMSDATA_H_INCLUDED

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EffectParamsDataNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class EffectParamsData : public EffectParamsDataNS::ConfigurableImpl<EffectParamsDataConfig>
	{
		// types
	public:
		typedef Parent Base;
		typedef EffectParamsData Parent;

		// constructors / destructors
	public:
		EXP_IMP explicit EffectParamsData( const EffectParamsDataConfig& cfg );
		EXP_IMP virtual ~EffectParamsData() = 0;
	
		// manipulation/ access
	public:

		// data
	private:

	};
}

#endif