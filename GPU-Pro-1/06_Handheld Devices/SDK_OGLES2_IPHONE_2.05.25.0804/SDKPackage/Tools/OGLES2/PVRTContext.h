/******************************************************************************

 @File         PVRTContext.h

 @Title        OGLES2/PVRTContext

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Context specific stuff - i.e. 3D API-related.

******************************************************************************/
#ifndef _PVRTCONTEXT_H_
#define _PVRTCONTEXT_H_


#include <stdio.h>
#if defined(__APPLE__)
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#else
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES2/gl2extimg.h>
#endif

/****************************************************************************
** Macros
****************************************************************************/
#define PVRTRGBA(r, g, b, a)   ((GLuint) (((a) << 24) | ((b) << 16) | ((g) << 8) | (r)))

/****************************************************************************
** Defines
****************************************************************************/


/****************************************************************************
** Enumerations
****************************************************************************/

/****************************************************************************
** Structures
****************************************************************************/
struct SPVRTContext
{
	int reserved;	// No context info for OGL.
};


#endif /* _PVRTCONTEXT_H_ */

/*****************************************************************************
 End of file (PVRTContext.h)
*****************************************************************************/
