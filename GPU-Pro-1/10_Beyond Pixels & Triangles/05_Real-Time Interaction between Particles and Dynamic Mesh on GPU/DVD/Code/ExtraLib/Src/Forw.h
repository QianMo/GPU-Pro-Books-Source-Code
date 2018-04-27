#ifndef EXTRALIB_FORW_H_INCLUDED
#define EXTRALIB_FORW_H_INCLUDED

namespace Mod
{
#include "DefDeclareClass.h"

	MOD_DECLARE_CLASS(TransformController)
	MOD_DECLARE_CLASS(EffectParamControllerCreator)
	MOD_DECLARE_CLASS(UserEffectParamVariables)
	MOD_DECLARE_CLASS(CameraController)
	MOD_DECLARE_CLASS(EntityEditor)
	MOD_DECLARE_CLASS(FPSCounter)
	
	struct TargetedCameraControllerConfig;

	MOD_DECLARE_NON_SHARED_CLASS(TransformControllerCreator)

	class VarVariant;

#include "UndefDeclareClass.h"	

}

#endif