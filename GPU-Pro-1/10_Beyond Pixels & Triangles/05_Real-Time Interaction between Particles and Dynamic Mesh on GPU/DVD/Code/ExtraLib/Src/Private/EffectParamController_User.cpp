#include "Precompiled.h"

#include "Common/Src/XIAttribute.h"
#include "Common/Src/TypedParam.h"

#include "SceneRender/Src/EffectParamControllerConfig.h"
#include "SceneRender/Src/EffectParam.h"

#include "UserEffectParamVariables.h"

#include "EffectParamController_User.h"

namespace Mod
{
	using namespace Math;

	//------------------------------------------------------------------------

	template< typename T>
	/*explicit*/
	EffectParamController_User<T>::EffectParamController_User( const EffectParamControllerConfig& cfg ) :
	Base ( cfg )
	{
		mValue = UserEffectParamVariables::Single().GetItem( XIAttString ( cfg.elem, L"variable" ) );		
	}

	//------------------------------------------------------------------------

	template< typename T>
	EffectParamController_User<T>::~EffectParamController_User()
	{

	}

	//------------------------------------------------------------------------

	template< typename T>
	/*virtual*/
	void
	EffectParamController_User<T>::UpdateImpl( float dt ) /*OVERRIDE*/
	{
		dt;
		GetConfig().param->SetValue( mValue->GetVal<T>() );
	}

	//------------------------------------------------------------------------

	EffectParamController_UserFloat::EffectParamController_UserFloat( const EffectParamControllerConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_UserFloat::~EffectParamController_UserFloat()
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_UserFloat2::EffectParamController_UserFloat2( const EffectParamControllerConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_UserFloat2::~EffectParamController_UserFloat2()
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_UserFloat3::EffectParamController_UserFloat3( const EffectParamControllerConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_UserFloat3::~EffectParamController_UserFloat3()
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_UserFloat4::EffectParamController_UserFloat4( const EffectParamControllerConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_UserFloat4::~EffectParamController_UserFloat4()
	{

	}

	//------------------------------------------------------------------------

	EffectParamControllerPtr CreateEffectParamController_UserFloat( const EffectParamControllerConfig& cfg )
	{
		return EffectParamControllerPtr( new EffectParamController_UserFloat( cfg ) );
	}

	//------------------------------------------------------------------------

	EffectParamControllerPtr CreateEffectParamController_UserFloat2( const EffectParamControllerConfig& cfg )
	{
		return EffectParamControllerPtr( new EffectParamController_UserFloat2( cfg ) );
	}

	//------------------------------------------------------------------------

	EffectParamControllerPtr CreateEffectParamController_UserFloat3( const EffectParamControllerConfig& cfg )
	{
		return EffectParamControllerPtr( new EffectParamController_UserFloat3( cfg ) );
	}

	//------------------------------------------------------------------------

	EffectParamControllerPtr CreateEffectParamController_UserFloat4( const EffectParamControllerConfig& cfg )
	{
		return EffectParamControllerPtr( new EffectParamController_UserFloat4( cfg ) );
	}



}