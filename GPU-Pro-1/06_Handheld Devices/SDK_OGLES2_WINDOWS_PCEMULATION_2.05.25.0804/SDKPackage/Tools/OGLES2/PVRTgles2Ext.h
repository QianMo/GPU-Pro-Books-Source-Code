/******************************************************************************

 @File         PVRTgles2Ext.h

 @Title        PVRTgles2Ext

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  OpenGL ES 2.0 extensions

******************************************************************************/
#ifndef _PVRTGLES2EXT_H_
#define _PVRTGLES2EXT_H_

#ifdef __APPLE__
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#ifdef TARGET_OS_IPHONE
// No binary shaders are allowed on the iphone and so this value is not defined
// Defining here allows for a more graceful fail of binary shader loading at runtime
// which can be recovered from instead of fail at compile time
#define GL_SGX_BINARY_IMG 0
#endif
#else
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES2/gl2extimg.h>
#endif

/****************************************************************************
** Build options
****************************************************************************/

#define GL_PVRTGLESEXT_VERSION 2

/**************************************************************************
****************************** GL EXTENSIONS ******************************
**************************************************************************/

class CPVRTgles2Ext
{

public:
    /* Type definitions for pointers to functions returned by eglGetProcAddress*/
    typedef void (GL_APIENTRY *PFNGLMULTIDRAWELEMENTS) (GLenum mode, GLsizei *count, GLenum type, const GLvoid **indices, GLsizei primcount); // glvoid
    typedef void* (GL_APIENTRY *PFNGLMAPBUFFEROES)(GLenum target, GLenum access);
    typedef GLboolean (GL_APIENTRY *PFNGLUNMAPBUFFEROES)(GLenum target);
    typedef void (GL_APIENTRY *PFNGLGETBUFFERPOINTERVOES)(GLenum target, GLenum pname, void** params);
	typedef void (GL_APIENTRY * PFNGLMULTIDRAWARRAYS) (GLenum mode, GLint *first, GLsizei *count, GLsizei primcount); // glvoid


	/* GL_EXT_multi_draw_arrays */
	PFNGLMULTIDRAWELEMENTS				glMultiDrawElementsEXT;
	PFNGLMULTIDRAWARRAYS				glMultiDrawArraysEXT;

	/* GL_EXT_multi_draw_arrays */
    PFNGLMAPBUFFEROES                   glMapBufferOES;
    PFNGLUNMAPBUFFEROES                 glUnmapBufferOES;
    PFNGLGETBUFFERPOINTERVOES           glGetBufferPointervOES;

public:
	/*!***********************************************************************
	@Function			LoadExtensions
	@Description		Initialises IMG extensions
	*************************************************************************/
	void LoadExtensions();

	/*!***********************************************************************
	@Function			IsGLExtensionSupported
	@Input				extension extension to query for
	@Returns			True if the extension is supported
	@Description		Queries for support of an extension
	*************************************************************************/
	static bool IsGLExtensionSupported(const char *extension);
};

#endif /* _PVRTGLES2EXT_H_ */

/*****************************************************************************
 End of file (PVRTgles2Ext.h)
*****************************************************************************/
