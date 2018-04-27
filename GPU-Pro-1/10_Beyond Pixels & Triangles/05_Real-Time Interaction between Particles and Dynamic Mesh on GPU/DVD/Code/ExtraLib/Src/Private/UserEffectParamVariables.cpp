#include "Precompiled.h"

#include "ContainerVeneers.h"

#include "UserEffectParamVariablesConfig.h"
#include "UserEffectParamVariables.h"

#define MD_NAMESPACE UserEffectParamVariablesNS
#include "ConfigurableImpl.cpp.h"

#define MD_NAMESPACE UserEffectParamVariablesNS
#include "Singleton.cpp.h"

namespace Mod
{
	template class UserEffectParamVariablesNS::Singleton<UserEffectParamVariables>;

	UserEffectParamVariables::UserEffectParamVariables( const UserEffectParamVariablesConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	UserEffectParamVariables::~UserEffectParamVariables()
	{

	}

	//------------------------------------------------------------------------
	
	ConstTypedParamPtr
	UserEffectParamVariables::GetItem( const String& key ) const
	{
		return map_get( mItemMap, key );
	}

	//------------------------------------------------------------------------

	void
	UserEffectParamVariables::RegisterParam( const String& key, ConstTypedParamPtr param )
	{
		MD_FERROR_ON_TRUE( map_has( mItemMap, key ) );
		mItemMap.insert( ItemMap::value_type( key, param ) );
	}

}