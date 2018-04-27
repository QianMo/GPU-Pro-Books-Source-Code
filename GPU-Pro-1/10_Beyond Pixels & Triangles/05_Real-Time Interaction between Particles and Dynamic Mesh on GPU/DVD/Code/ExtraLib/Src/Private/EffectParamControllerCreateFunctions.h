#ifndef EXTRALIB_EFFECTPARAMCONTROLLERCREATEFUNCTIONS_H_INCLUDED
#define EXTRALIB_EFFECTPARAMCONTROLLERCREATEFUNCTIONS_H_INCLUDED

namespace Mod
{
	EffectParamControllerPtr CreateEffectParamController_AnimatedTexture( const EffectParamControllerConfig& cfg );

	EffectParamControllerPtr CreateEffectParamController_LinearFloat( const EffectParamControllerConfig& cfg );
	EffectParamControllerPtr CreateEffectParamController_LinearFloat2( const EffectParamControllerConfig& cfg );
	EffectParamControllerPtr CreateEffectParamController_LinearFloat3( const EffectParamControllerConfig& cfg );
	EffectParamControllerPtr CreateEffectParamController_LinearFloat4( const EffectParamControllerConfig& cfg );

	EffectParamControllerPtr CreateEffectParamController_UserFloat( const EffectParamControllerConfig& cfg );
	EffectParamControllerPtr CreateEffectParamController_UserFloat2( const EffectParamControllerConfig& cfg );
	EffectParamControllerPtr CreateEffectParamController_UserFloat3( const EffectParamControllerConfig& cfg );
	EffectParamControllerPtr CreateEffectParamController_UserFloat4( const EffectParamControllerConfig& cfg );
}

#endif