#include "Precompiled.h"

#include "CameraControllerConfig.h"
#include "CameraController.h"

#define MD_NAMESPACE CameraControllerNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	template class CameraControllerNS::ConfigurableImpl<CameraControllerConfig>;

	/*explicit*/
	CameraController::CameraController( const CameraControllerConfig& cfg ) :
	Base ( cfg )
	{

	}

	//------------------------------------------------------------------------

	CameraController::~CameraController()
	{

	}

	//------------------------------------------------------------------------
	
	void
	CameraController::Update( float dt )
	{
		UpdateImpl( dt * GetConfig().speed );
	}
}