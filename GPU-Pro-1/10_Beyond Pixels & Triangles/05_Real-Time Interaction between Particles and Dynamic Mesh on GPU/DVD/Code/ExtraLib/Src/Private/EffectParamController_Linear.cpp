#include "Precompiled.h"

#include "Common/Src/XIElemAttribute.h"

#include "Math/Src/Operations.h"

#include "SceneRender/Src/EffectParamControllerConfig.h"
#include "SceneRender/Src/EffectParam.h"

#include "EffectParamController_Linear.h"

namespace Mod
{
	using namespace Math;

	namespace
	{
		void FillValue( float& oVal, const String& name, const XMLElemPtr& elem )
		{
			oVal = XIFloat( elem, name, L"val" );
		}

		void FillValue( float2& oVal, const String& name, const XMLElemPtr& elem )
		{
			oVal.x = XIFloat( elem, name, L"x" );
			oVal.y = XIFloat( elem, name, L"y" );
		}

		void FillValue( float3& oVal, const String& name, const XMLElemPtr& elem )
		{
			oVal.x = XIFloat( elem, name, L"x" );
			oVal.y = XIFloat( elem, name, L"y" );
			oVal.z = XIFloat( elem, name, L"z" );
		}

		void FillValue( float4& oVal, const String& name, const XMLElemPtr& elem )
		{
			oVal.x = XIFloat( elem, name, L"x" );
			oVal.y = XIFloat( elem, name, L"y" );
			oVal.z = XIFloat( elem, name, L"z" );
			oVal.w = XIFloat( elem, name, L"w" );
		}
	}

	//------------------------------------------------------------------------

	template< typename T>
	/*explicit*/
	EffectParamController_Linear<T>::EffectParamController_Linear( const EffectParamControllerConfig& cfg ) :
	Base ( cfg )
	{
		FillValue( mStartVal, L"start", cfg.elem );
		FillValue( mEndVal, L"end", cfg.elem );
		FillValue( mSpeed, L"speed", cfg.elem );
	}

	//------------------------------------------------------------------------

	template< typename T>
	EffectParamController_Linear<T>::~EffectParamController_Linear()
	{

	}

	//------------------------------------------------------------------------

	template< typename T>
	/*virtual*/
	void
	EffectParamController_Linear<T>::UpdateImpl( float dt ) /*OVERRIDE*/
	{
		mTime += dt * mSpeed;

		mTime = fmodf( mTime, 1.0f );

		GetConfig().param->SetValue( lerp( mStartVal, mEndVal, mTime ) );
	}

	//------------------------------------------------------------------------

	EffectParamController_LinearFloat::EffectParamController_LinearFloat( const EffectParamControllerConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_LinearFloat::~EffectParamController_LinearFloat()
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_LinearFloat2::EffectParamController_LinearFloat2( const EffectParamControllerConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_LinearFloat2::~EffectParamController_LinearFloat2()
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_LinearFloat3::EffectParamController_LinearFloat3( const EffectParamControllerConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_LinearFloat3::~EffectParamController_LinearFloat3()
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_LinearFloat4::EffectParamController_LinearFloat4( const EffectParamControllerConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	EffectParamController_LinearFloat4::~EffectParamController_LinearFloat4()
	{

	}

	//------------------------------------------------------------------------

	EffectParamControllerPtr CreateEffectParamController_LinearFloat( const EffectParamControllerConfig& cfg )
	{
		return EffectParamControllerPtr( new EffectParamController_LinearFloat( cfg ) );
	}

	//------------------------------------------------------------------------

	EffectParamControllerPtr CreateEffectParamController_LinearFloat2( const EffectParamControllerConfig& cfg )
	{
		return EffectParamControllerPtr( new EffectParamController_LinearFloat2( cfg ) );
	}

	//------------------------------------------------------------------------

	EffectParamControllerPtr CreateEffectParamController_LinearFloat3( const EffectParamControllerConfig& cfg )
	{
		return EffectParamControllerPtr( new EffectParamController_LinearFloat3( cfg ) );
	}

	//------------------------------------------------------------------------

	EffectParamControllerPtr CreateEffectParamController_LinearFloat4( const EffectParamControllerConfig& cfg )
	{
		return EffectParamControllerPtr( new EffectParamController_LinearFloat4( cfg ) );
	}



}