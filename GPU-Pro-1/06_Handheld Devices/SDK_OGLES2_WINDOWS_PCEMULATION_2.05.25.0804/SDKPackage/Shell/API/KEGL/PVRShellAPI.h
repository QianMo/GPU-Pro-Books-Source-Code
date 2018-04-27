/******************************************************************************

 @File         PVRShellAPI.h

 @Title        KEGL/PVRShellAPI

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
#ifdef BUILD_OGLES2
	#include <GLES2/gl2.h>
	#include <EGL/egl.h>
#elif BUILD_OGL
#define SUPPORT_OPENGL
#if defined(WIN32) || defined(UNDER_CE)
	#include <windows.h>
#endif
	#include <GL/gl.h>
	#include <EGL/egl.h>
#elif BUILD_OVG
#include <VG/openvg.h>
#include <EGL/egl.h>
#else
	#include <GLES/egl.h>
#endif



/*!***************************************************************************
 @Class PVRShellInitAPI
 @Brief Initialisation interface with specific API.
****************************************************************************/
class PVRShellInitAPI
{
public:
	EGLDisplay	gEglDisplay;
	EGLSurface	gEglWindow;
	EGLContext	gEglContext;
	EGLConfig	gEglConfig;
	EGLint		majorVersion, minorVersion;
	bool		powerManagementSupported;

public:
	EGLConfig SelectEGLConfiguration(const PVRShellData * const pData);
	const char *StringFrom_eglGetError() const;
};

#endif // __PVRSHELLAPI_H_

/*****************************************************************************
 End of file (PVRShellAPI.h)
*****************************************************************************/
