/******************************************************************************

 @File         PVRShellAPI.h

 @Title        EAGL/PVRShellAPI

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Makes programming for 3D APIs easier by wrapping surface
               initialization, Texture allocation and other functions for use by a demo.

******************************************************************************/
#ifndef __PVRSHELLAPI_H_
#define __PVRSHELLAPI_H_

/****************************************************************************
** 3D API header files
****************************************************************************/
#ifdef BUILD_OGLES
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>
#elif BUILD_OGLES2
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#endif

/*!***************************************************************************
 @Class PVRShellInitAPI
 @Brief Initialisation interface with specific API.
****************************************************************************/
class PVRShellInitAPI
{

};
#endif // __PVRSHELLAPI_H_

/*****************************************************************************
 End of file (PVRShellAPI.h)
*****************************************************************************/
