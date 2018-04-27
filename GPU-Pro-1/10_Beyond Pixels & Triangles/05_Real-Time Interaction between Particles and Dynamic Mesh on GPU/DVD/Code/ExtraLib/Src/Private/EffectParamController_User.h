#ifndef EXTRALIB_EFFECTPARAMCONTROLLER_ANIMATEDTEXTURE_H_INCLUDED
#define EXTRALIB_EFFECTPARAMCONTROLLER_ANIMATEDTEXTURE_H_INCLUDED


#include "Common/Src/Forw.h"

#include "Math/Src/Types.h"

#include "SceneRender/Src/EffectParamController.h"

#include "Forw.h"

namespace Mod
{
	template< typename T>
	class EffectParamController_User : public EffectParamController
	{
		// types
	public:
		typedef Types< ShaderResourcePtr > :: Vec	ShaderResourceVec;
		typedef Parent Base;
		typedef EffectParamController_User Parent;

		// contruction/ destruction
	public:
		explicit EffectParamController_User( const EffectParamControllerConfig& cfg );
		~EffectParamController_User();

		// polymorpism
	private:
		virtual void UpdateImpl( float dt ) OVERRIDE;

		// data
	private:
		ConstTypedParamPtr mValue;
	};

	//------------------------------------------------------------------------

	class EffectParamController_UserFloat : public EffectParamController_User< float >
	{
	public:
		EffectParamController_UserFloat( const EffectParamControllerConfig& cfg );
		~EffectParamController_UserFloat();

	private:

	};

	//------------------------------------------------------------------------

	class EffectParamController_UserFloat2 : public EffectParamController_User< Math::float2 >
	{
	public:
		EffectParamController_UserFloat2( const EffectParamControllerConfig& cfg );
		~EffectParamController_UserFloat2();

	private:

	};

	//------------------------------------------------------------------------

	class EffectParamController_UserFloat3 : public EffectParamController_User< Math::float3 >
	{
	public:
		EffectParamController_UserFloat3( const EffectParamControllerConfig& cfg );
		~EffectParamController_UserFloat3();

	private:

	};

	//------------------------------------------------------------------------

	class EffectParamController_UserFloat4 : public EffectParamController_User< Math::float4 >
	{
	public:
		EffectParamController_UserFloat4( const EffectParamControllerConfig& cfg );
		~EffectParamController_UserFloat4();

	private:

	};

}

#endif