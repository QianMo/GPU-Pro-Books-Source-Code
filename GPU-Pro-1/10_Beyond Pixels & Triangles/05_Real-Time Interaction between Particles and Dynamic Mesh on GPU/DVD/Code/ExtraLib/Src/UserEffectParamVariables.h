#ifndef EXTRALIB_USEREFFECTPARAMVARIABLES_H_INCLUDED
#define EXTRALIB_USEREFFECTPARAMVARIABLES_H_INCLUDED

#include "Common/Src/Forw.h"

#include "Forw.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE UserEffectParamVariablesNS
#include "ConfigurableImpl.h"

#define MD_NAMESPACE UserEffectParamVariablesNS
#include "Singleton.h"

namespace Mod
{

	class UserEffectParamVariables :	public UserEffectParamVariablesNS::ConfigurableImpl<UserEffectParamVariablesConfig>,
										public UserEffectParamVariablesNS::Singleton<UserEffectParamVariables>
	{
		// types
	public:
		typedef Types2< String, ConstTypedParamPtr > :: Map ItemMap;

		// constructors / destructors
	public:
		explicit UserEffectParamVariables( const UserEffectParamVariablesConfig& cfg );
		~UserEffectParamVariables();
	
		// manipulation/ access
	public:
		ConstTypedParamPtr	GetItem( const String& key ) const;
		void				RegisterParam( const String& key, ConstTypedParamPtr param );

		// data
	private:
		ItemMap		mItemMap;

	};
}

#endif