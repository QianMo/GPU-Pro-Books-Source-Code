/******************************************************************************

 @File         PVRTShader.cpp

 @Title        PVRTShader

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Shader handling for OpenGL ES 2.0

******************************************************************************/
#include "PVRTString.h"
#include "PVRTShader.h"
#include "PVRTResourceFile.h"

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
											CPVRTString* const pReturnError)
{
	/* Create and compile the shader object */
    *pObject = glCreateShader(Type);
    glShaderSource(*pObject, 1, &pszShaderCode, NULL);
    glCompileShader(*pObject);

	/* Test if compilation succeeded */
	GLint ShaderCompiled;
    glGetShaderiv(*pObject, GL_COMPILE_STATUS, &ShaderCompiled);
	if (!ShaderCompiled)
	{
		int i32InfoLogLength, i32CharsWritten;
		glGetShaderiv(*pObject, GL_INFO_LOG_LENGTH, &i32InfoLogLength);
		char* pszInfoLog = new char[i32InfoLogLength];
        glGetShaderInfoLog(*pObject, i32InfoLogLength, &i32CharsWritten, pszInfoLog);
		*pReturnError = CPVRTString("Failed to compile shader: ") + pszInfoLog + "\n";
		delete [] pszInfoLog;
		glDeleteShader(*pObject);
		return PVR_FAIL;
	}

	return PVR_SUCCESS;
}

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
EPVRTError PVRTShaderLoadBinaryFromMemory(	const void* const ShaderData,
											const size_t Size,
											const GLenum Type,
											const GLenum Format,
											GLuint* const pObject,
											CPVRTString* const pReturnError)
{
	/* Create and compile the shader object */
    *pObject = glCreateShader(Type);

    // Get the list of supported binary formats
    // and if (more then 0) find given Format among them
    GLint numFormats = 0;
    GLint *listFormats;
    int i;
    glGetIntegerv(GL_NUM_SHADER_BINARY_FORMATS,&numFormats);
    if(numFormats != 0) {
        listFormats = new GLint[numFormats];
        for(i=0;i<numFormats;++i)
            listFormats[i] = 0;
        glGetIntegerv(GL_SHADER_BINARY_FORMATS,listFormats);
        for(i=0;i<numFormats;++i) {
            if(listFormats[i] == Format) {
                glShaderBinary(1, pObject, Format, ShaderData, (GLint)Size);
                if (glGetError() != GL_NO_ERROR)
                {
                    *pReturnError = CPVRTString("Failed to load binary shader\n");
                    glDeleteShader(*pObject);
                    return PVR_FAIL;
                }
                return PVR_SUCCESS;
            }
        }
        delete [] listFormats;
    }
    *pReturnError = CPVRTString("Failed to load binary shader\n");
    glDeleteShader(*pObject);
    return PVR_FAIL;
}

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
									const SPVRTContext* const pContext)
{
	*pReturnError = "";

	if(Format)
	{
		CPVRTResourceFile ShaderFile(pszBinFile);
		if (ShaderFile.IsOpen())
		{
			if(PVRTShaderLoadBinaryFromMemory(ShaderFile.DataPtr(), ShaderFile.Size(), Type, Format, pObject, pReturnError) == PVR_SUCCESS)
				return PVR_SUCCESS;
		}

		*pReturnError += CPVRTString("Failed to open shader ") + pszBinFile + "\n";
	}

	CPVRTResourceFile ShaderFile(pszSrcFile);
	if (!ShaderFile.IsOpen())
	{
		*pReturnError += CPVRTString("Failed to open shader ") + pszSrcFile + "\n";
		return PVR_FAIL;
	}

	return PVRTShaderLoadSourceFromMemory(ShaderFile.StringPtr(), Type, pObject, pReturnError);
}


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
								CPVRTString* const pReturnError)
{
	*pProgramObject = glCreateProgram();

    glAttachShader(*pProgramObject, FragmentShader);
    glAttachShader(*pProgramObject, VertexShader);

	for (int i = 0; i < i32NumAttribs; ++i)
	{
		glBindAttribLocation(*pProgramObject, i, pszAttribs[i]);
	}

	// Link the program object
    glLinkProgram(*pProgramObject);
    GLint Linked;
    glGetProgramiv(*pProgramObject, GL_LINK_STATUS, &Linked);
	if (!Linked)
	{
		int i32InfoLogLength, i32CharsWritten;
		glGetProgramiv(*pProgramObject, GL_INFO_LOG_LENGTH, &i32InfoLogLength);
		char* pszInfoLog = new char[i32InfoLogLength];
		glGetProgramInfoLog(*pProgramObject, i32InfoLogLength, &i32CharsWritten, pszInfoLog);
		*pReturnError = CPVRTString("Failed to link: ") + pszInfoLog + "\n";
		delete [] pszInfoLog;
		return PVR_FAIL;
	}

	glUseProgram(*pProgramObject);

	return PVR_SUCCESS;
}

/*****************************************************************************
 End of file (PVRTShader.cpp)
*****************************************************************************/
