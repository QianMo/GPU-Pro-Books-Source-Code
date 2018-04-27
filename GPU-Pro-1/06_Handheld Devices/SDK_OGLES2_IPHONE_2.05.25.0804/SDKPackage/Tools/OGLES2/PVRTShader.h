/******************************************************************************

 @File         PVRTShader.h

 @Title        OGLES2/PVRTShader

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Shader handling for OpenGL ES 2.0

******************************************************************************/
#ifndef _PVRTSHADER_H_
#define _PVRTSHADER_H_

#include "PVRTContext.h"
#include "../PVRTString.h"
#include "../PVRTError.h"

/*!***************************************************************************
 @Function		PVRTShaderLoadSourceFromMemory
 @Input			pszShaderCode	shader source code
 @Input			Type			type of shader (GL_VERTEX_SHADER or GL_FRAGMENT_SHADER)
 @Output		pObject			the resulting shader object
 @Output		pReturnError	the error message if it failed
 @Return		PVR_SUCCESS on success and PVR_FAIL on failure (also fills the str string)
 @Description	Loads a shader source code into memory and compiles it.
*****************************************************************************/
EPVRTError PVRTShaderLoadSourceFromMemory(	const char* pszShaderCode,
											const GLenum Type,
											GLuint* const pObject,
											CPVRTString* const pReturnError);

/*!***************************************************************************
 @Function		PVRTShaderLoadBinaryFromMemory
 @Input			ShaderData		shader compiled binary data
 @Input			Size			size of shader binary data in bytes
 @Input			Type			type of shader (GL_VERTEX_SHADER or GL_FRAGMENT_SHADER)
 @Input			Format			shader binary format
 @Output		pObject			the resulting shader object
 @Output		pReturnError	the error message if it failed
 @Return		PVR_SUCCESS on success and PVR_FAIL on failure (also fills the str string)
 @Description	Takes a shader binary from memory and passes it to the GL.
*****************************************************************************/
EPVRTError PVRTShaderLoadBinaryFromMemory(	const void*  const ShaderData,
											const size_t Size,
											const GLenum Type,
											const GLenum Format,
											GLuint*  const pObject,
											CPVRTString*  const pReturnError);

/*!***************************************************************************
 @Function		PVRTShaderLoadFromFile
 @Input			pszBinFile		binary shader filename
 @Input			pszSrcFile		source shader filename
 @Input			Type			type of shader (GL_VERTEX_SHADER or GL_FRAGMENT_SHADER)
 @Input			Format			shader binary format, or 0 for source shader
 @Output		pObject			the resulting shader object
 @Output		pReturnError	the error message if it failed
 @Input			pContext		Context
 @Return		PVR_SUCCESS on success and PVR_FAIL on failure (also fills pReturnError)
 @Description	Loads a shader file into memory and passes it to the GL.
*****************************************************************************/
EPVRTError PVRTShaderLoadFromFile(	const char* const pszBinFile,
									const char* const pszSrcFile,
									const GLenum Type,
									const GLenum Format,
									GLuint* const pObject,
									CPVRTString* const pReturnError,
									const SPVRTContext* const pContext=0);

/*!***************************************************************************
 @Function		PVRTCreateProgram
 @Output		pProgramObject			the created program object
 @Input			VertexShader			the vertex shader to link
 @Input			FragmentShader			the fragment shader to link
 @Input			pszAttribs				an array of attribute names
 @Input			i32NumAttribs			the number of attributes to bind
 @Output		pReturnError			the error message if it failed
 @Returns		PVR_SUCCESS on success, PVR_FAIL if failure
 @Description	Links a shader program.
*****************************************************************************/
EPVRTError PVRTCreateProgram(	GLuint* const pProgramObject,
								const GLuint VertexShader,
								const GLuint FragmentShader,
								const char** const pszAttribs,
								const int i32NumAttribs,
								CPVRTString* const pReturnError);

#endif

/*****************************************************************************
 End of file (PVRTShader.h)
*****************************************************************************/
