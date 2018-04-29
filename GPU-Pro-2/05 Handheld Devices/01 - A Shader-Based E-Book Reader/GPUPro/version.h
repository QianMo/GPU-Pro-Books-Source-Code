/******************************************************************************

 @File         version.h

 @Title        EBook Demo

 @Copyright    Copyright (C) 2010 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  

******************************************************************************/

#ifndef _VERSION_H_
#define _VERSION_H_

#ifdef BUILD_OGLES2
#define API		"ES2.0"
#else
#define API		"GL2.0"
#endif

/* History Log
V0.0.8
- Added Print3D
- Improved Render on Demand (fills all frame buffers)
V0.0.7
- Code improvements and documentation
V0.0.6
- Render on Demand
- #defines in shaders
V0.0.5
- Support for Rotation
- Improved edge shaders
V0.0.4
- Arithmetic optimisations in shaders
- Some operations moved to CPU
V0.0.3
- Added Book class to handle multiple pages
V0.0.2
- Added constraints to prevent page tearing
V0.0.1
- Added double page fold (implemented through derived classes and virtuals)



*/

#define MAJOR	"0"
#define MINOR	"0"
#define FIX		"8"
#define VERSION_NUMBER API "-V" MAJOR "." MINOR "." FIX
#define VERSION_NOAPI (MAJOR "." MINOR "." FIX)

#endif
