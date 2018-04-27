#ifndef EXTRALIB_CAMERACONTROLLERCONFIG_H_INCLUDED
#define EXTRALIB_CAMERACONTROLLERCONFIG_H_INCLUDED

#include "SceneRender/Src/Forw.h"

namespace Mod
{

	struct CameraControllerConfig
	{
		CameraPtr	camera;
		float		speed;

		CameraControllerConfig();
		virtual ~CameraControllerConfig();

		virtual CameraControllerConfig* Clone() const = 0;
	};

	struct TargetedCameraControllerConfig : CameraControllerConfig
	{
		ENodePtr target;

		virtual TargetedCameraControllerConfig* Clone() const OVERRIDE;
	};

#ifdef _MSC_VER
	// unquellable warning forced us to quell it like this
	namespace
	{
		CameraControllerConfig* (CameraControllerConfig::*MD_MSVC_ERROR_ClonePtr )() const = &CameraControllerConfig::Clone;
	}
#endif
	

}

#endif