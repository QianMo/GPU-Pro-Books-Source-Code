#ifndef EXTRALIB_TRANSFORMCONTROLLERCREATEFUNCTIONS_H_INCLUDED
#define EXTRALIB_TRANSFORMCONTROLLERCREATEFUNCTIONS_H_INCLUDED

#include "Forw.h"

#include "Common/Src/Forw.h"

namespace Mod
{
	TransformControllerPtr CreateTransformController_SpindleMove( const TransformControllerConfig& cfg );
	TransformControllerPtr CreateTransformController_SpindleWobble( const TransformControllerConfig& cfg );

}

#endif