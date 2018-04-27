/******************************************************************************

 @File         PVRTPFXParserAPI.cpp

 @Title        PVRTPFXParserAPI

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  PFX file parser.

******************************************************************************/
#include <stdio.h>
#include <string.h>

#if !defined(_UITRON_) && !defined(__SYMBIAN32__) && !defined(__APPLE__)
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#include "PVRTContext.h"
#include "PVRTMatrix.h"
#include "PVRTFixedPoint.h"
#include "PVRTString.h"
#include "PVRTShader.h"
#include "PVRTPFXParser.h"
#include "PVRTPFXParserAPI.h"
#include "PVRTTexture.h"
#include "PVRTgles2Ext.h"

/*!***************************************************************************
 @Function			CPVRTPFXEffect Constructor
 @Description		Sets the context and initialises the member variables to zero.
*****************************************************************************/
CPVRTPFXEffect::CPVRTPFXEffect()
{
	m_psContext = NULL;
	m_uiProgram = 0;
	m_pnTextureIdx = 0;
}

/*!***************************************************************************
 @Function			CPVRTPFXEffect Constructor
 @Description		Sets the context and initialises the member variables to zero.
*****************************************************************************/
CPVRTPFXEffect::CPVRTPFXEffect(SPVRTContext &sContext)
{
	m_psContext = &sContext;
	m_uiProgram = 0;
	m_pnTextureIdx = 0;
}

/*!***************************************************************************
 @Function			CPVRTPFXEffect Destructor
 @Description		Calls Destroy().
*****************************************************************************/
CPVRTPFXEffect::~CPVRTPFXEffect()
{
	Destroy();
}

/*!***************************************************************************
 @Function			Load
 @Input				src					PFX Parser Object
 @Input				pszEffect			Effect name
 @Input				pszFileName			Effect file name
 @Output			pReturnError		Error string
 @Returns			EPVRTError			PVR_SUCCESS if load succeeded
 @Description		Loads the specified effect from the CPVRTPFXParser object.
					Compiles and links the shaders. Initialises texture data.
*****************************************************************************/
EPVRTError CPVRTPFXEffect::Load(CPVRTPFXParser &src, const char * const pszEffect, const char * const pszFileName, CPVRTString *pReturnError)
{
	GLuint				uiVertexShader, uiFragShader;
	unsigned int		i, j;

	if(!src.m_nNumEffects)
		return PVR_FAIL;

	/*
		First find the named effect from the effect file
	*/
	if(pszEffect)
	{
		for(i = 0; i < src.m_nNumEffects; ++i)
		{
			if(strcmp(src.m_psEffect[i].pszName, pszEffect) == 0)
			{
				m_nEffect = i;
				break;
			}
		}
		if(i == src.m_nNumEffects)
		{
			return PVR_FAIL;
		}
	}
	else
	{
		m_nEffect = 0;
	}

	/*
		Now load the effect
	*/
	m_pParser = &src;
	SPVRTPFXParserEffect *psParserEffect = &src.m_psEffect[m_nEffect];

	// Create room for per-texture data
	m_psTextures = new SPVRTPFXTexture[src.m_nNumTextures];

	// Initialise each Texture
	for(i = 0; i < src.m_nNumTextures; ++i)
	{
		m_psTextures[i].p	= src.m_psTexture[i].pszFile;
		m_psTextures[i].ui	= 0xFFFFFFFF;
	}

	// Initialise the effect
	{
		// initialise attributes to default values
		char *pszVertexShader = NULL;
		char *pszFragmentShader = NULL;
		bool bFreeVertexShader = false;
		bool bFreeFragmentShader = false;

		// find shaders requested
		for(i=0; i < src.m_nNumVertShaders; ++i)
		{
			if(strcmp(psParserEffect->pszVertexShaderName, src.m_psVertexShader[i].pszName) == 0)
			{
                if(src.m_psVertexShader[i].bUseFileName)
				{
					pszVertexShader = src.m_psVertexShader[i].pszGLSLcode;
				}
				else
				{
					// offset glsl code by nFirstLineNumber
					pszVertexShader = (char *)malloc((strlen(src.m_psVertexShader[i].pszGLSLcode) + (src.m_psVertexShader[i].nFirstLineNumber) + 1) * sizeof(char));
					pszVertexShader[0] = '\0';
					for(unsigned int n = 0; n < src.m_psVertexShader[i].nFirstLineNumber; n++)
						strcat(pszVertexShader, "\n");
					strcat(pszVertexShader, src.m_psVertexShader[i].pszGLSLcode);

					bFreeVertexShader = true;
				}

				break;
			}
		}
		for(i=0; i<src.m_nNumFragShaders; ++i)
		{
			if(strcmp(psParserEffect->pszFragmentShaderName, src.m_psFragmentShader[i].pszName) == 0)
			{
                if(src.m_psFragmentShader[i].bUseFileName)
				{
					pszFragmentShader = src.m_psFragmentShader[i].pszGLSLcode;
				}
				else
				{
					// offset glsl code by nFirstLineNumber
					pszFragmentShader = (char *)malloc((strlen(src.m_psFragmentShader[i].pszGLSLcode) + (src.m_psFragmentShader[i].nFirstLineNumber) + 1) * sizeof(char));
					pszFragmentShader[0] = '\0';
					for(unsigned int n = 0; n < src.m_psFragmentShader[i].nFirstLineNumber; n++)
						strcat(pszFragmentShader, "\n");
					strcat(pszFragmentShader, src.m_psFragmentShader[i].pszGLSLcode);

					bFreeFragmentShader = true;
				}

				break;
			}
		}

		CPVRTString error;
		bool		bLoadSource = 1;

		// Try first to load from the binary block
		if (src.m_psVertexShader[i].pbGLSLBinary!=NULL)
		{
			if (PVRTShaderLoadBinaryFromMemory(src.m_psVertexShader[i].pbGLSLBinary, src.m_psVertexShader[i].nGLSLBinarySize,
												GL_VERTEX_SHADER, GL_SGX_BINARY_IMG, &uiVertexShader, &error) == PVR_SUCCESS)
			{
				// success loading the binary block so we do not need to load the source
				bLoadSource = 0;
			}
			else
			{
				bLoadSource = 1;
			}
		}

		// If it fails, load from source
		if (bLoadSource)
		{
			if(pszVertexShader)
			{
				if (PVRTShaderLoadSourceFromMemory(pszVertexShader, GL_VERTEX_SHADER, &uiVertexShader, &error) != PVR_SUCCESS)
				{
					*pReturnError = CPVRTString("Vertex Shader compile error in file '") + pszFileName + "':\n" + error;
					if(bFreeVertexShader)	FREE(pszVertexShader);
					if(bFreeFragmentShader)	FREE(pszFragmentShader);
					return PVR_FAIL;
				}
			}
			else // Shader not found or failed binary block
			{
				if (src.m_psVertexShader[i].pbGLSLBinary==NULL)
				{
					*pReturnError = CPVRTString("Vertex shader ") + psParserEffect->pszVertexShaderName + "  not found in " + pszFileName + ".\n";
				}
				else
				{
					*pReturnError = CPVRTString("Binary vertex shader ") + psParserEffect->pszVertexShaderName + " not supported.\n";
				}

				if(bFreeVertexShader)	FREE(pszVertexShader);
				if(bFreeFragmentShader)	FREE(pszFragmentShader);
				return PVR_FAIL;
			}
		}

		// Try first to load from the binary block
		if (src.m_psFragmentShader[i].pbGLSLBinary!=NULL)
		{
			if (PVRTShaderLoadBinaryFromMemory(src.m_psFragmentShader[i].pbGLSLBinary, src.m_psVertexShader[i].nGLSLBinarySize,
													GL_FRAGMENT_SHADER, GL_SGX_BINARY_IMG, &uiFragShader, &error) == PVR_SUCCESS)
			{
				// success loading the binary block so we do not need to load the source
				bLoadSource = 0;
			}
			else
			{
				bLoadSource = 1;
			}
		}

		// If it fails, load from source
		if (bLoadSource)
		{
			if(pszFragmentShader)
			{
				if (PVRTShaderLoadSourceFromMemory(pszFragmentShader, GL_FRAGMENT_SHADER, &uiFragShader, &error) != PVR_SUCCESS)
				{
					*pReturnError = CPVRTString("Fragment Shader compile error in file '") + pszFileName + "':\n" + error;
					if(bFreeVertexShader)	FREE(pszVertexShader);
					if(bFreeFragmentShader)	FREE(pszFragmentShader);
					return PVR_FAIL;
				}
			}
			else // Shader not found or failed binary block
			{
				if (src.m_psFragmentShader[i].pbGLSLBinary==NULL)
				{
					*pReturnError = CPVRTString("Fragment shader ") + psParserEffect->pszFragmentShaderName + "  not found in " + pszFileName + ".\n";
				}
				else
				{
					*pReturnError = CPVRTString("Binary Fragment shader ") + psParserEffect->pszFragmentShaderName + " not supported.\n";
				}

				if(bFreeVertexShader)
					FREE(pszVertexShader);
				if(bFreeFragmentShader)
					FREE(pszFragmentShader);

				return PVR_FAIL;
			}
		}

		if(bFreeVertexShader)
			FREE(pszVertexShader);

		if(bFreeFragmentShader)
			FREE(pszFragmentShader);

		// Create the shader program
		m_uiProgram = glCreateProgram();


		// Attach the fragment and vertex shaders to it
		glAttachShader(m_uiProgram, uiFragShader);
		glAttachShader(m_uiProgram, uiVertexShader);

		glDeleteShader(uiVertexShader);
		glDeleteShader(uiFragShader);

		// Bind vertex attributes
		for(i = 0; i < psParserEffect->nNumAttributes; ++i)
		{
			glBindAttribLocation(m_uiProgram, i, psParserEffect->psAttribute[i].pszName);
		}

		//	Link the program.
		glLinkProgram(m_uiProgram);
		GLint Linked;
		glGetProgramiv(m_uiProgram, GL_LINK_STATUS, &Linked);
		if (!Linked)
		{
			int i32InfoLogLength, i32CharsWritten;
			glGetProgramiv(m_uiProgram, GL_INFO_LOG_LENGTH, &i32InfoLogLength);
			char* pszInfoLog = new char[i32InfoLogLength];
			glGetProgramInfoLog(m_uiProgram, i32InfoLogLength, &i32CharsWritten, pszInfoLog);
			*pReturnError = CPVRTString("Error Linking shaders in file '") + pszFileName + "':\n\n"
							+ CPVRTString("Failed to link: ") + pszInfoLog + "\n";
			delete [] pszInfoLog;
			return PVR_FAIL;
		}

		/*
			Textures
		*/
		m_pnTextureIdx = new unsigned int[psParserEffect->nNumTextures];
		for(i = 0; i < psParserEffect->nNumTextures; ++i)
		{
			for(j = 0; j < src.m_nNumTextures; ++j)
			{
				if(strcmp(psParserEffect->psTextures[i].pszName, src.m_psTexture[j].pszName) == 0)
				{
					m_pnTextureIdx[i] = j;
					break;
				}
			}
			if(j == src.m_nNumTextures)
			{
				*pReturnError = CPVRTString("Effect \"") +  psParserEffect->pszName + "\", requested non-existent texture: \""
								+ psParserEffect->psTextures[i].pszName + "\"\n";
				m_pnTextureIdx[i] = 0;
			}
		}
	}

	return PVR_SUCCESS;
}

/*!***************************************************************************
 @Function			Destroy
 @Description		Deletes the gl program object and texture data.
*****************************************************************************/
void CPVRTPFXEffect::Destroy()
{
	{
		if(m_uiProgram != 0)
		{
			glDeleteProgram(m_uiProgram);
			m_uiProgram = 0;
		}

		delete [] m_pnTextureIdx;
		m_pnTextureIdx = 0;
	}

	delete [] m_psTextures;
	m_psTextures = 0;
}

/*!***************************************************************************
 @Function			Activate
 @Returns			PVR_SUCCESS if activate succeeded
 @Description		Selects the gl program object and binds the textures.
*****************************************************************************/
EPVRTError CPVRTPFXEffect::Activate()
{
	unsigned int i;
	SPVRTPFXParserEffect *psParserEffect = &m_pParser->m_psEffect[m_nEffect];

	// Set the program
	glUseProgram(m_uiProgram);

	// Set the textures
	for(i = 0; i < psParserEffect->nNumTextures; ++i)
	{
		glActiveTexture(GL_TEXTURE0 + psParserEffect->psTextures[i].nNumber);
		if((psParserEffect->psTextures[m_pnTextureIdx[i]].u32Type&PVRTEX_CUBEMAP)!=0)
			glBindTexture(GL_TEXTURE_CUBE_MAP, m_psTextures[m_pnTextureIdx[i]].ui);
		else
			glBindTexture(GL_TEXTURE_2D, m_psTextures[m_pnTextureIdx[i]].ui);
	}

	return PVR_SUCCESS;
}

/*!***************************************************************************
 @Function			GetSemantics
 @Output			psUniforms				pointer to application uniform data array
 @Output			pnUnknownUniformCount	unknown uniform count
 @Input				psParams				pointer to semantic data array
 @Input				nParamCount				number of samantic items
 @Input				psUniformSemantics		pointer to uniform semantics array
 @Input				nUniformSemantics		number of uniform semantic items
 @Input				pglesExt				opengl extensions object
 @Input				uiProgram				program object index
 @Input				bIsAttribue				true if getting attribute semantics
 @Output			errorMsg				error string
 @Returns			unsigned int			number of successful semantics
 @Description		Get the data array for the semantics.
*****************************************************************************/
static unsigned int GetSemantics(
	SPVRTPFXUniform					* const psUniforms,
	unsigned int					* const pnUnknownUniformCount,
	const SPVRTPFXParserSemantic	* const psParams,
	const unsigned int				nParamCount,
	const SPVRTPFXUniformSemantic	* const psUniformSemantics,
	const unsigned int				nUniformSemantics,
	const GLuint					uiProgram,
	bool							bIsAttribue,
	CPVRTString						* const errorMsg)
{
	unsigned int	i, j, nCount, nCountUnused;
	int				nLocation;

	/*
		Loop over the parameters searching for their semantics. If
		found/recognised, it should be placed in the output array.
	*/
	nCount = 0;
	nCountUnused = 0;

	for(j = 0; j < nParamCount; ++j)
	{
		for(i = 0; i < nUniformSemantics; ++i)
		{
			if(strcmp(psParams[j].pszValue, psUniformSemantics[i].p) != 0)
			{
				continue;
			}

			// Semantic found for this parameter
			if(bIsAttribue)
			{
				nLocation = glGetAttribLocation(uiProgram, psParams[j].pszName);
			}
			else
			{
				nLocation = glGetUniformLocation(uiProgram, psParams[j].pszName);
			}

			if(nLocation != -1)
			{
				if(psUniforms)
				{
					psUniforms[nCount].nSemantic	= psUniformSemantics[i].n;
					psUniforms[nCount].nLocation	= nLocation;
					psUniforms[nCount].nIdx			= psParams[j].nIdx;
				}
				++nCount;
			}
			else
			{
				*errorMsg += "WARNING: Variable not used by GLSL code: ";
				*errorMsg += CPVRTString(psParams[j].pszName) + " ";
				*errorMsg += CPVRTString(psParams[j].pszValue) + "\n";
				++nCountUnused;
			}

			// Skip to the next parameter
			break;
		}
		if(i == nUniformSemantics)
		{
			*errorMsg += "WARNING: Semantic unknown to application: ";
			*errorMsg += CPVRTString(psParams[j].pszName) + " ";
			*errorMsg += CPVRTString(psParams[j].pszValue) + "\n";
		}
	}

	*pnUnknownUniformCount	= nParamCount - nCount - nCountUnused;
	return nCount;
}

/*!***************************************************************************
 @Function			BuildUniformTable
 @Output			ppsUniforms					pointer to uniform data array
 @Output			pnUniformCount				pointer to number of uniforms
 @Output			pnUnknownUniformCount		pointer to number of unknown uniforms
 @Input				psUniformSemantics			pointer to uniform semantic data array
 @Input				nSemantics					number of uniform semantics
 @Output			pReturnError				error string
 @Returns			EPVRTError					PVR_SUCCESS if succeeded
 @Description		Builds the uniform table from the semantics.
*****************************************************************************/
EPVRTError CPVRTPFXEffect::BuildUniformTable(
	SPVRTPFXUniform					** const ppsUniforms,
	unsigned int					* const pnUniformCount,
	unsigned int					* const pnUnknownUniformCount,
	const SPVRTPFXUniformSemantic	* const psUniformSemantics,
	const unsigned int				nSemantics,
	CPVRTString							*pReturnError)
{
	unsigned int			nCount, nUnknownCount;
	SPVRTPFXUniform			*psUniforms;
	SPVRTPFXParserEffect	*psParserEffect			= &m_pParser->m_psEffect[m_nEffect];

	nCount = 0;
	nCount += GetSemantics(NULL, &nUnknownCount, psParserEffect->psUniform, psParserEffect->nNumUniforms, psUniformSemantics, nSemantics, m_uiProgram, false, pReturnError);
	nCount += GetSemantics(NULL, &nUnknownCount, psParserEffect->psAttribute, psParserEffect->nNumAttributes, psUniformSemantics, nSemantics, m_uiProgram, true, pReturnError);

	psUniforms = (SPVRTPFXUniform*)malloc(nCount * sizeof(*psUniforms));
	if(!psUniforms)
		return PVR_FAIL;

	*pReturnError = "";

	nCount = 0;
	nCount += GetSemantics(&psUniforms[nCount], &nUnknownCount, psParserEffect->psUniform, psParserEffect->nNumUniforms, psUniformSemantics, nSemantics, m_uiProgram, false, pReturnError);
	*pnUnknownUniformCount	= nUnknownCount;

	nCount += GetSemantics(&psUniforms[nCount], &nUnknownCount, psParserEffect->psAttribute, psParserEffect->nNumAttributes, psUniformSemantics, nSemantics, m_uiProgram, true, pReturnError);
	*pnUnknownUniformCount	+= nUnknownCount;

	*ppsUniforms			= psUniforms;
	*pnUniformCount			= nCount;
	return PVR_SUCCESS;
}

/*!***************************************************************************
 @Function			GetTextureArray
 @Output			nCount					number of textures
 @Returns			SPVRTPFXTexture*		pointer to the texture data array
 @Description		Gets the texture data array.
*****************************************************************************/
const SPVRTPFXTexture *CPVRTPFXEffect::GetTextureArray(unsigned int &nCount) const
{
	nCount = m_pParser->m_nNumTextures;
	return m_psTextures;
}

/*!***************************************************************************
 @Function			SetTexture
 @Input				nIdx				texture number
 @Input				ui					opengl texture handle
 @Input				u32flags			texture flags
 @Description		Sets the textrue and applys the filtering.
*****************************************************************************/
void CPVRTPFXEffect::SetTexture(const unsigned int nIdx, const GLuint ui, const unsigned int u32flags)
{
	if(nIdx < m_pParser->m_nNumTextures)
	{
		GLenum u32Target = GL_TEXTURE_2D;

		// Check if texture is a cubemap
		if((u32flags & PVRTEX_CUBEMAP) != 0)
			u32Target = GL_TEXTURE_CUBE_MAP;

		// Set default filter from PFX file
		switch(m_pParser->m_psTexture[nIdx].nMIP)
		{
		case 0:
			switch(m_pParser->m_psTexture[nIdx].nMin)
			{
			case 0:
				glTexParameteri(u32Target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				break;
			case 1:
				glTexParameteri(u32Target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				break;
			}
			break;
		case 1:
			switch(m_pParser->m_psTexture[nIdx].nMin)
			{
			case 0:
				glTexParameteri(u32Target, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
				break;
			case 1:
				glTexParameteri(u32Target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
				break;
			}
			break;
		case 2:
			switch(m_pParser->m_psTexture[nIdx].nMin)
			{
			case 0:
				glTexParameteri(u32Target, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
				break;
			case 1:
				glTexParameteri(u32Target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				break;
			}
			break;
		}

		switch(m_pParser->m_psTexture[nIdx].nMag)
		{
		case 0:
			glTexParameteri(u32Target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			break;
		case 1:
			glTexParameteri(u32Target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			break;
		}

		switch(m_pParser->m_psTexture[nIdx].nWrapS)
		{
		case 0:
			glTexParameteri(u32Target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			break;
		case 1:
			glTexParameteri(u32Target, GL_TEXTURE_WRAP_S, GL_REPEAT);
			break;
		}

		switch(m_pParser->m_psTexture[nIdx].nWrapT)
		{
		case 0:
			glTexParameteri(u32Target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			break;
		case 1:
			glTexParameteri(u32Target, GL_TEXTURE_WRAP_T, GL_REPEAT);
			break;
		}

#ifdef GL_TEXTURE_WRAP_R
		switch(m_pParser->m_psTexture[nIdx].nWrapR)
		{
		case 0:
			glTexParameteri(u32Target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
			break;
		case 1:
			glTexParameteri(u32Target, GL_TEXTURE_WRAP_R, GL_REPEAT);
			break;
		}
#endif

		m_psTextures[nIdx].ui = ui;

		// store flags
		m_pParser->m_psEffect[m_nEffect].psTextures[nIdx].u32Type = u32flags;
	}
}


/*!***************************************************************************
 @Function			SetDefaultSemanticValue
 @Input				pszName				name of uniform
 @Input				psDefaultValue      pointer to default value
 @Description		Sets the dafault value for the uniform semantic.
*****************************************************************************/
void CPVRTPFXEffect::SetDefaultUniformValue(const char *const pszName, const SPVRTSemanticDefaultData *psDefaultValue)
{
	GLint nLocation = glGetUniformLocation(m_uiProgram, pszName);

	switch(psDefaultValue->eType)
	{
		case eDataTypeMat2:
			glUniformMatrix2fv(nLocation, 1, GL_FALSE, psDefaultValue->pfData);
			break;
		case eDataTypeMat3:
			glUniformMatrix3fv(nLocation, 1, GL_FALSE, psDefaultValue->pfData);
			break;
		case eDataTypeMat4:
			glUniformMatrix4fv(nLocation, 1, GL_FALSE, psDefaultValue->pfData);
			break;
		case eDataTypeVec2:
			glUniform2fv(nLocation, 1, psDefaultValue->pfData);
			break;
		case eDataTypeVec3:
			glUniform3fv(nLocation, 1, psDefaultValue->pfData);
			break;
		case eDataTypeVec4:
			glUniform4fv(nLocation, 1, psDefaultValue->pfData);
			break;
		case eDataTypeIvec2:
			glUniform2iv(nLocation, 1, psDefaultValue->pnData);
			break;
		case eDataTypeIvec3:
			glUniform3iv(nLocation, 1, psDefaultValue->pnData);
			break;
		case eDataTypeIvec4:
			glUniform4iv(nLocation, 1, psDefaultValue->pnData);
			break;
		case eDataTypeBvec2:
			glUniform2i(nLocation, psDefaultValue->pbData[0] ? 1 : 0, psDefaultValue->pbData[1] ? 1 : 0);
			break;
		case eDataTypeBvec3:
			glUniform3i(nLocation, psDefaultValue->pbData[0] ? 1 : 0, psDefaultValue->pbData[1] ? 1 : 0, psDefaultValue->pbData[2] ? 1 : 0);
			break;
		case eDataTypeBvec4:
			glUniform4i(nLocation, psDefaultValue->pbData[0] ? 1 : 0, psDefaultValue->pbData[1] ? 1 : 0, psDefaultValue->pbData[2] ? 1 : 0, psDefaultValue->pbData[3] ? 1 : 0);
			break;
		case eDataTypeFloat:
			glUniform1f(nLocation, psDefaultValue->pfData[0]);
			break;
		case eDataTypeInt:
			glUniform1i(nLocation, psDefaultValue->pnData[0]);
			break;
		case eDataTypeBool:
			glUniform1i(nLocation, psDefaultValue->pbData[0] ? 1 : 0);
			break;

		case eNumDefaultDataTypes:
		case eDataTypeNone:
		default:
			break;
	}
}

/*****************************************************************************
 End of file (PVRTPFXParserAPI.cpp)
*****************************************************************************/
