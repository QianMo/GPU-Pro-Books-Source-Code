#include "Precompiled.h"

#include "EffectVariationProvider.h"

#include "Providers.h"

#include "EffectVariationMapConfig.h"
#include "EffectVariationMap.h"

#define MD_NAMESPACE EffectVariationMapNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	template class EffectVariationMapNS::ConfigurableImpl<EffectVariationMapConfig>;

	//------------------------------------------------------------------------

	EffectVariationMap::EffectVariationMap( const EffectVariationMapConfig& cfg ) : 
	Parent( cfg ),
	mLastID( EffectVariationID(-1) )
	{
		EffectVariationID defID = GetVariationIDByName( L"Default" );
		MD_FERROR_ON_FALSE( defID == EVI::DEFAULT_VAR_ID );
	}

	//------------------------------------------------------------------------

	EffectVariationMap::~EffectVariationMap()
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectVariationID
	EffectVariationMap::GetVariationIDByName( const String& name )
	{
		VariationMap::const_iterator  found = mVariationMap.find( name );
		if( found != mVariationMap.end() )
			return found->second;
		else
		{
			EffectVariationPtr var =  Providers::Single().GetEffectVariationProv()->GetItem( name );
			mVariationMap[ name ] = ++mLastID;
			mVariationTable.resize( mLastID + 1 );
			mVariationTable[ mLastID ] = var;
		}

		return mLastID;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectVariationPtr
	EffectVariationMap::GetVariationByID( EffectVariationID id ) const
	{
		MD_FERROR_ON_FALSE( id < mVariationTable.size() );
		return mVariationTable[ id ];
	}

	//------------------------------------------------------------------------


}