#include "Precompiled.h"

#include "TransformControllerCreateFunctions.h"

#include "TransformController_SpindleMove.h"
#include "TransformController_SpindleWobble.h"

namespace Mod
{
	TransformControllerPtr
	CreateTransformController_SpindleMove( const TransformControllerConfig& cfg )
	{
		return TransformControllerPtr( new TransformController_SpindleMove( cfg ) );
	}

	//------------------------------------------------------------------------

	TransformControllerPtr
	CreateTransformController_SpindleWobble( const TransformControllerConfig& cfg )
	{
		return TransformControllerPtr( new TransformController_SpindleWobble( cfg ) );
	}

}