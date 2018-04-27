#ifndef EXTRALIB_EFFECTPARAMCONTROLLER_LINEAR_H_INCLUDED
#define EXTRALIB_EFFECTPARAMCONTROLLER_LINEAR_H_INCLUDED

#include "Forw.h"

#include "Math/Src/Types.h"

#include "Wrap3D/Src/Forw.h"

#include "SceneRender/Src/EffectParamController.h"

namespace Mod
{
	template< typename T>
	class EffectParamController_Linear : public EffectParamController
	{
		// types
	public:
		typedef Types< ShaderResourcePtr > :: Vec	ShaderResourceVec;
		typedef Parent Base;
		typedef EffectParamController_Linear Parent;

		// contruction/ destruction
	public:
		explicit EffectParamController_Linear( const EffectParamControllerConfig& cfg );
		~EffectParamController_Linear();

		// polymorpism
	private:
		virtual void UpdateImpl( float dt ) OVERRIDE;

		// data
	private:
		float 				mSpeed;
		float				mTime;
		T					mStartVal;
		T					mEndVal;
	};

	//------------------------------------------------------------------------

	class EffectParamController_LinearFloat : public EffectParamController_Linear< float >
	{
	public:
		EffectParamController_LinearFloat( const EffectParamControllerConfig& cfg );
		~EffectParamController_LinearFloat();

	private:

	};

	//------------------------------------------------------------------------

	class EffectParamController_LinearFloat2 : public EffectParamController_Linear< Math::float2 >
	{
	public:
		EffectParamController_LinearFloat2( const EffectParamControllerConfig& cfg );
		~EffectParamController_LinearFloat2();

	private:

	};

	//------------------------------------------------------------------------

	class EffectParamController_LinearFloat3 : public EffectParamController_Linear< Math::float3 >
	{
	public:
		EffectParamController_LinearFloat3( const EffectParamControllerConfig& cfg );
		~EffectParamController_LinearFloat3();

	private:

	};

	//------------------------------------------------------------------------

	class EffectParamController_LinearFloat4 : public EffectParamController_Linear< Math::float4 >
	{
	public:
		EffectParamController_LinearFloat4( const EffectParamControllerConfig& cfg );
		~EffectParamController_LinearFloat4();

	private:

	};

}

#endif