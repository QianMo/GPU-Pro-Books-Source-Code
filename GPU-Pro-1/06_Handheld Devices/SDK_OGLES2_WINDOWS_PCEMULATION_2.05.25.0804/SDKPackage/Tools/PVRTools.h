/******************************************************************************

 @File         PVRTools.h

 @Title        PVRTools

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Header file to include a particular API tools header

******************************************************************************/
#ifndef PVRTOOLS_H
#define PVRTOOLS_H

#ifdef BUILD_OGLES2
	#include "OGLES2Tools.h"
#elif BUILD_OGLES
	#include "OGLESTools.h"
#elif BUILD_OGL
	#include "OGLTools.h"
#elif BUILD_D3DM
	#include "D3DMTools.h"
#elif BUILD_DX9
	#include "DX9Tools.h"
#endif

#endif /* PVRTOOLS_H*/

/*****************************************************************************
 End of file (Tools.h)
*****************************************************************************/
