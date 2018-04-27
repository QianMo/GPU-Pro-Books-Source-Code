#include "Precompiled.h"

#include "CameraControllerConfig.h"

namespace Mod
{
	//------------------------------------------------------------------------
	CameraControllerConfig::CameraControllerConfig() :
	speed( 1.f )
	{

	}

	//------------------------------------------------------------------------
	/*virtual*/

	CameraControllerConfig::~CameraControllerConfig()
	{

	}

	//------------------------------------------------------------------------

	/*virtual*/
	TargetedCameraControllerConfig* TargetedCameraControllerConfig::Clone() const /*OVERRIDE*/
	{
		return new TargetedCameraControllerConfig( * this );
	}
}