/******************************************************************************

 @File         PVRTPrint3DAPI.cpp

 @Title        OGLES2/PVRTPrint3DAPI

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Displays a text string using 3D polygons. Can be done in two ways:
               using a window defined by the user or writing straight on the
               screen.

******************************************************************************/
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "PVRTContext.h"
#include "PVRTFixedPoint.h"
#include "PVRTMatrix.h"
#include "PVRTTexture.h"
#include "PVRTTextureAPI.h"
#include "PVRTPrint3D.h"
#include "PVRTString.h"
#include "PVRTShader.h"
#include "PVRTgles2Ext.h"

#include "PVRTPrint3DShaders.h"

/****************************************************************************
** Defines
****************************************************************************/
#define VERTEX_ARRAY			0
#define UV_ARRAY				1
#define COLOR_ARRAY				2

/****************************************************************************
** Structures
****************************************************************************/

struct SPVRTPrint3DAPI
{
	GLuint						uTexture[5];
	GLuint						uTexturePVRLogo;
	GLuint						uTextureIMGLogo;

	GLuint						m_VertexShaderObject, m_FragmentShaderObject;
	GLuint						m_ProgramObject;

/* Used to save the OpenGL state to restore them after drawing */
	GLboolean					isCullFaceEnabled;
	GLboolean					isBlendEnabled;
	GLboolean					isDepthTestEnabled;
	GLint						nArrayBufferBinding;
};

/****************************************************************************
** Class: CPVRTPrint3D
****************************************************************************/

/*!***************************************************************************
 @Function			ReleaseTextures
 @Description		Deallocate the memory allocated in SetTextures(...)
*****************************************************************************/
void CPVRTPrint3D::ReleaseTextures()
{
#if !defined (DISABLE_PRINT3D)

	if(m_pAPI)
	{
		/* Release the shaders */
		glDeleteProgram(m_pAPI->m_ProgramObject);
		glDeleteShader(m_pAPI->m_VertexShaderObject);
		glDeleteShader(m_pAPI->m_FragmentShaderObject);
	}

	/* Only release textures if they've been allocated */
	if (!m_bTexturesSet) return;

	/* Release IndexBuffer */
	FREE(m_pwFacesFont);
	FREE(m_pPrint3dVtx);

	/* Delete textures */
	glDeleteTextures(5, m_pAPI->uTexture);
	glDeleteTextures(1, &m_pAPI->uTexturePVRLogo);
	glDeleteTextures(1, &m_pAPI->uTextureIMGLogo);

	m_bTexturesSet = false;

	FREE(m_pVtxCache);

	APIRelease();

#endif
}

/*!***************************************************************************
 @Function			Flush
 @Description		Flushes all the print text commands
*****************************************************************************/
int CPVRTPrint3D::Flush()
{
#if !defined (DISABLE_PRINT3D)

	int		nTris, nVtx, nVtxBase, nTrisTot;

	_ASSERT((m_nVtxCache % 4) == 0);
	_ASSERT(m_nVtxCache <= m_nVtxCacheMax);

	/* Save render states */
	APIRenderStates(0);

	/* Set font texture */
	glBindTexture(GL_TEXTURE_2D, m_pAPI->uTexture[0]);

	nTrisTot = m_nVtxCache >> 1;

	/*
		Render the text then. Might need several submissions.
	*/
	nVtxBase = 0;
	while(m_nVtxCache)
	{
		nVtx	= PVRT_MIN(m_nVtxCache, 0xFFFC);
		nTris	= nVtx >> 1;

		_ASSERT(nTris <= (PVRTPRINT3D_MAX_RENDERABLE_LETTERS*2));
		_ASSERT((nVtx % 4) == 0);

		/* Draw triangles */
		glVertexAttribPointer(VERTEX_ARRAY, 3, GL_FLOAT, GL_FALSE, sizeof(SPVRTPrint3DAPIVertex), (const void*)&m_pVtxCache[nVtxBase].sx);
		glVertexAttribPointer(COLOR_ARRAY, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(SPVRTPrint3DAPIVertex), (const void*)&m_pVtxCache[nVtxBase].color);
		glVertexAttribPointer(UV_ARRAY, 2, GL_FLOAT, GL_FALSE, sizeof(SPVRTPrint3DAPIVertex), (const void*)&m_pVtxCache[nVtxBase].tu);
		glDrawElements(GL_TRIANGLES, nTris * 3, GL_UNSIGNED_SHORT, m_pwFacesFont);
		if (glGetError())
		{
			_RPT0(_CRT_WARN,"glDrawElements(GL_TRIANGLES, (VertexCount/2)*3, GL_UNSIGNED_SHORT, m_pFacesFont); failed\n");
		}

		nVtxBase		+= nVtx;
		m_nVtxCache	-= nVtx;
	}

	/* Draw a logo if requested */
#if defined(FORCE_NO_LOGO)
	/* Do nothing */

#elif defined(FORCE_PVR_LOGO)
    APIDrawLogo(ePVRTPrint3DLogoPVR, 1);	/* PVR to the right */

#elif defined(FORCE_IMG_LOGO)
	APIDrawLogo(ePVRTPrint3DLogoIMG, 1);	/* IMG to the right */

#elif defined(FORCE_ALL_LOGOS)
	APIDrawLogo(ePVRTPrint3DLogoIMG, -1); /* IMG to the left */
	APIDrawLogo(ePVRTPrint3DLogoPVR, 1);	/* PVR to the right */

#else
	/* User selected logos */
	switch (m_uLogoToDisplay)
	{
		case ePVRTPrint3DLogoNone:
			break;
		default:
		case ePVRTPrint3DLogoPVR:
			APIDrawLogo(ePVRTPrint3DLogoPVR, 1);	/* PVR to the right */
			break;
		case ePVRTPrint3DLogoIMG:
			APIDrawLogo(ePVRTPrint3DLogoIMG, 1);	/* IMG to the right */
			break;
		case (ePVRTPrint3DLogoPVR | ePVRTPrint3DLogoIMG):
			APIDrawLogo(ePVRTPrint3DLogoIMG, -1); /* IMG to the left */
			APIDrawLogo(ePVRTPrint3DLogoPVR, 1);	/* PVR to the right */
			break;
	}
#endif

	/* Restore render states */
	APIRenderStates(1);

	return nTrisTot;

#else
	return 0;
#endif
}

/*************************************************************
*					 PRIVATE FUNCTIONS						 *
**************************************************************/

/*!***************************************************************************
 @Function			APIInit
 @Description		Initialisation and texture upload. Should be called only once
					for a given context.
*****************************************************************************/
bool CPVRTPrint3D::APIInit(const SPVRTContext	* const pContext)
{
	m_pAPI = new SPVRTPrint3DAPI;
	if(!m_pAPI)
		return false;

	/* Compiles the shaders. For a more detailed explanation, see IntroducingPVRTools */
	CPVRTString error;
	bool bRes;
	// Try binary shaders first
	bRes = (PVRTShaderLoadBinaryFromMemory(_Print3DFragShader_fsc, _Print3DFragShader_fsc_size,
				GL_FRAGMENT_SHADER, GL_SGX_BINARY_IMG, &m_pAPI->m_FragmentShaderObject, &error) == PVR_SUCCESS)
	       && (PVRTShaderLoadBinaryFromMemory(_Print3DVertShader_vsc, _Print3DVertShader_vsc_size,
				GL_VERTEX_SHADER, GL_SGX_BINARY_IMG, &m_pAPI->m_VertexShaderObject, &error) == PVR_SUCCESS);
	if (!bRes)
	{
		// if binary shaders don't work, try source shaders
		bRes = (PVRTShaderLoadSourceFromMemory(_Print3DFragShader_fsh, GL_FRAGMENT_SHADER, &m_pAPI->m_FragmentShaderObject, &error) == PVR_SUCCESS) &&
			   (PVRTShaderLoadSourceFromMemory(_Print3DVertShader_vsh, GL_VERTEX_SHADER, &m_pAPI->m_VertexShaderObject, &error)  == PVR_SUCCESS);
	}
	_ASSERT(bRes);

    m_pAPI->m_ProgramObject = glCreateProgram();
    glAttachShader(m_pAPI->m_ProgramObject, m_pAPI->m_VertexShaderObject);
    glAttachShader(m_pAPI->m_ProgramObject, m_pAPI->m_FragmentShaderObject);
    glBindAttribLocation(m_pAPI->m_ProgramObject, VERTEX_ARRAY, "myVertex");
    glBindAttribLocation(m_pAPI->m_ProgramObject, UV_ARRAY, "myUV");
    glBindAttribLocation(m_pAPI->m_ProgramObject, COLOR_ARRAY, "myColour");

	glLinkProgram(m_pAPI->m_ProgramObject);
	GLint Linked;
	glGetProgramiv(m_pAPI->m_ProgramObject, GL_LINK_STATUS, &Linked);
	if (!Linked)
	{
		bRes = false;
	}

	_ASSERT(bRes);

	return true;
}

/*!***************************************************************************
 @Function			APIRelease
 @Description		Deinitialization.
*****************************************************************************/
void CPVRTPrint3D::APIRelease()
{
	delete m_pAPI;
	m_pAPI = 0;
}

/*!***************************************************************************
 @Function			APIUpLoadIcons
 @Description		Initialisation and texture upload. Should be called only once
					for a given context.
*****************************************************************************/
bool CPVRTPrint3D::APIUpLoadIcons(
	const PVRTuint32 * const pPVR,
	const PVRTuint32 * const pIMG)
{
	/* Load Icon textures */
	if(PVRTTextureLoadFromPointer((unsigned char*)pPVR, &m_pAPI->uTexturePVRLogo) != PVR_SUCCESS)
		return false;
	if(PVRTTextureLoadFromPointer((unsigned char*)pIMG, &m_pAPI->uTextureIMGLogo) != PVR_SUCCESS)
		return false;

	return true;
}

/*!***************************************************************************
 @Function			APIUpLoad4444
 @Return			true if succesful, false otherwise.
 @Description		Reads texture data from *.dat and loads it in
					video memory.
*****************************************************************************/
bool CPVRTPrint3D::APIUpLoad4444(unsigned int dwTexID, unsigned char *pSource, unsigned int nSize, unsigned int nMode)
{
	int				i, j;
	int				x=256, y=256;
	unsigned short	R, G, B, A;
	unsigned short	*p8888,  *pDestByte;
	unsigned char   *pSrcByte;

	/* Only square textures */
	x = nSize;
	y = nSize;

	glGenTextures(1, &m_pAPI->uTexture[dwTexID]);

	/* Load texture from data */

	/* Format is 4444-packed, expand it into 8888 */
	if (nMode==0)
	{
		/* Allocate temporary memory */
		p8888 = (unsigned short *)malloc(nSize*nSize*sizeof(unsigned short));
		if(!p8888)
		{
			PVRTErrorOutputDebug("Not enough memory!\n");
			return false;
		}

		pDestByte = p8888;

		/* Set source pointer (after offset of 16) */
		pSrcByte = &pSource[16];

		/* Transfer data */
		for (i=0; i<y; i++)
		{
			for (j=0; j<x; j++)
			{
				/* Get all 4 colour channels (invert A) */
				G =   (*pSrcByte) & 0xF0;
				R = ( (*pSrcByte++) & 0x0F ) << 4;
				A =   (*pSrcByte) ^ 0xF0;
				B = ( (*pSrcByte++) & 0x0F ) << 4;

				/* Set them in 8888 data */
				*pDestByte++ = ((R&0xF0)<<8) | ((G&0xF0)<<4) | (B&0xF0) | (A&0xF0)>>4;
			}
		}
	}
	else
	{
		/* Set source pointer */
		pSrcByte = pSource;

		/* Allocate temporary memory */
		p8888 = (unsigned short *)malloc(nSize*nSize*sizeof(unsigned short));
		if(!p8888)
		{
			PVRTErrorOutputDebug("Not enough memory!\n");
			return false;
		}


		/* Set destination pointer */
		pDestByte = p8888;

		/* Transfer data */
		for (i=0; i<y; i++)
		{
			for (j=0; j<x; j++)
			{
				/* Get alpha channel */
				A = *pSrcByte++;

				/* Set them in 8888 data */
				R = 255;
				G = 255;
				B = 255;

				/* Set them in 8888 data */
				*pDestByte++ = ((R&0xF0)<<8) | ((G&0xF0)<<4) | (B&0xF0) | (A&0xF0)>>4;
			}
		}
	}

	/* Bind texture */
	glBindTexture(GL_TEXTURE_2D, m_pAPI->uTexture[dwTexID]);

	/* Default settings: bilinear */
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	/* Now load texture */
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, x, y, 0, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4, p8888);
	if (glGetError())
	{
		_RPT0(_CRT_WARN,"glTexImage2D() failed\n");
		free(p8888);
		return false;
	}

	/* Destroy temporary data */
	free(p8888);

	/* Return status : OK */
	return true;
}

/*!***************************************************************************
 @Function			DrawBackgroundWindowUP
 @Description
*****************************************************************************/
void CPVRTPrint3D::DrawBackgroundWindowUP(unsigned int dwWin, SPVRTPrint3DAPIVertex *pVtx, const bool bIsOp, const bool bBorder)
{
	const unsigned short c_pwFacesWindow[] =
	{
		0,1,2, 2,1,3, 2,3,4, 4,3,5, 4,5,6, 6,5,7, 5,8,7, 7,8,9, 8,10,9, 9,10,11, 8,12,10, 8,13,12,
		13,14,12, 13,15,14, 13,3,15, 1,15,3, 3,13,5, 5,13,8
	};

	/* Set the texture (with or without border) */
	if(!bBorder)
		glBindTexture(GL_TEXTURE_2D, m_pAPI->uTexture[2 + (bIsOp*2)]);
	else
		glBindTexture(GL_TEXTURE_2D, m_pAPI->uTexture[1 + (bIsOp*2)]);

	/* Is window opaque ? */
	if(bIsOp)
	{
		glDisable(GL_BLEND);
	}
	else
	{
		/* Set blending properties */
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	/* Set pointers */
	glVertexAttribPointer(VERTEX_ARRAY, 3, GL_FLOAT, GL_FALSE, sizeof(SPVRTPrint3DAPIVertex), (const void*)&pVtx[0].sx);
	glVertexAttribPointer(COLOR_ARRAY, 3, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(SPVRTPrint3DAPIVertex), (const void*)&pVtx[0].color);
	glVertexAttribPointer(UV_ARRAY, 2, GL_FLOAT, GL_FALSE, sizeof(SPVRTPrint3DAPIVertex), (const void*)&pVtx[0].tu);

	/* Draw triangles */
	glDrawElements(GL_TRIANGLES, 18*3, GL_UNSIGNED_SHORT, c_pwFacesWindow);
	if (glGetError())
	{
		PVRTErrorOutputDebug("glDrawElements(GL_TRIANGLES, 18*3, GL_UNSIGNED_SHORT, pFaces); failed\n");
	}

	/* Restore render states (need to be translucent to draw the text) */
}

/*!***************************************************************************
 @Function			APIRenderStates
 @Description		Stores, writes and restores Render States
*****************************************************************************/
void CPVRTPrint3D::APIRenderStates(int nAction)
{
//	static GLboolean	bVertexPointerEnabled, bColorPointerEnabled, bTexCoorPointerEnabled;
	PVRTMATRIX			Matrix;
	int					i;

	/* Saving or restoring states ? */
	switch (nAction)
	{
	case 0:
	{
		/* Get previous render states */
		m_pAPI->isCullFaceEnabled = glIsEnabled(GL_CULL_FACE);
		m_pAPI->isBlendEnabled = glIsEnabled(GL_BLEND);
		m_pAPI->isDepthTestEnabled = glIsEnabled(GL_DEPTH_TEST);
		glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&m_pAPI->nArrayBufferBinding);

		/******************************
		** SET PRINT3D RENDER STATES **
		******************************/

		/* Set the default GL_ARRAY_BUFFER */
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/* Get viewport dimensions */
		/*glGetFloatv(GL_VIEWPORT, fViewport);*/

		/* Set matrix with viewport dimensions */
		for(i=0; i<16; i++)
		{
			Matrix.f[i]=0;
		}
		Matrix.f[0] =	(2.0f/(m_fScreenScale[0]*640.0f));
		Matrix.f[5] =	(-2.0f/(m_fScreenScale[1]*480.0f));
		Matrix.f[10] = (1.0f);
		Matrix.f[12] = (-1.0f);
		Matrix.f[13] = (1.0f);
		Matrix.f[15] = (1.0f);

		/* Use the shader */
		glUseProgram(m_pAPI->m_ProgramObject);

		/* Bind the projection and modelview matrices to the shader */
		int location = glGetUniformLocation(m_pAPI->m_ProgramObject, "myMVPMatrix");
		glUniformMatrix4fv( location, 1, GL_FALSE, Matrix.f);

		/* Culling */
		glEnable(GL_CULL_FACE);
		glFrontFace(GL_CW);
		glCullFace(GL_FRONT);

		/* Set blending mode */
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		/* Set Z compare properties */
		glDisable(GL_DEPTH_TEST);

		/* Set client states */
		glEnableVertexAttribArray(VERTEX_ARRAY);
		glEnableVertexAttribArray(COLOR_ARRAY);
		glEnableVertexAttribArray(UV_ARRAY);

		/* texture 	*/
		glActiveTexture(GL_TEXTURE1);
		glActiveTexture(GL_TEXTURE0);
		break;
	}
	case 1:
		/* Restore render states */
		glDisableVertexAttribArray(VERTEX_ARRAY);
		glDisableVertexAttribArray(COLOR_ARRAY);
		glDisableVertexAttribArray(UV_ARRAY);

		/* Restore some values */
		if (!m_pAPI->isCullFaceEnabled) glDisable(GL_CULL_FACE);
		if (!m_pAPI->isBlendEnabled) glDisable(GL_BLEND);
		if (m_pAPI->isDepthTestEnabled) glEnable(GL_DEPTH_TEST);
		glBindBuffer(GL_ARRAY_BUFFER,m_pAPI->nArrayBufferBinding);

		break;
	}
}

/****************************************************************************
** Local code
****************************************************************************/

/*!***************************************************************************
 @Function			APIDrawLogo
 @Description		nPos = -1 to the left
					nPos = +1 to the right
*****************************************************************************/
#define LOGO_SIZE 0.3f
#define LOGO_SHIFT 0.05f

void CPVRTPrint3D::APIDrawLogo(unsigned int uLogoToDisplay, int nPos)
{
	const float fLogoSizeHalf = 0.15f;
	const float fLogoShift = 0.05f;
	const float fLogoSizeHalfShifted = fLogoSizeHalf + fLogoShift;
	const float fLogoYScale = 50.0f / 64.0f;

	static VERTTYPE	Vertices[] =
		{
			-fLogoSizeHalf, fLogoSizeHalf , 0.5f,
			-fLogoSizeHalf, -fLogoSizeHalf, 0.5f,
			fLogoSizeHalf , fLogoSizeHalf , 0.5f,
	 		fLogoSizeHalf , -fLogoSizeHalf, 0.5f
		};

	static float	Colours[] = {
			(1.0f), (1.0f), (1.0f), (0.75f),
			(1.0f), (1.0f), (1.0f), (0.75f),
			(1.0f), (1.0f), (1.0f), (0.75f),
	 		(1.0f), (1.0f), (1.0f), (0.75f)
		};

	static float	UVs[] = {
			(0.0f), (0.0f),
			(0.0f), (1.0f),
			(1.0f), (0.0f),
	 		(1.0f), (1.0f)
		};

	float *pVertices = ( (float*)&Vertices );
	float *pColours  = ( (float*)&Colours );
	float *pUV       = ( (float*)&UVs );
	GLuint	tex;

	switch(uLogoToDisplay)
	{
	case ePVRTPrint3DLogoIMG:
		tex = m_pAPI->uTextureIMGLogo;
		break;
	default:
		tex = m_pAPI->uTexturePVRLogo;
		break;
	}

	// Matrices
	PVRTMATRIX matModelView;
	PVRTMATRIX matTransform;
	PVRTMatrixIdentity(matModelView);

	float fScreenScale = PVRT_MIN(m_ui32ScreenDim[0], m_ui32ScreenDim[1]) / 480.0f;
	float fScaleX = (640.0f / m_ui32ScreenDim[0]) * fScreenScale;
	float fScaleY = (480.0f / m_ui32ScreenDim[1]) * fScreenScale * fLogoYScale;

	PVRTMatrixScaling(matTransform, f2vt(fScaleX), f2vt(fScaleY), f2vt(1.0f));
	PVRTMatrixMultiply(matModelView, matModelView, matTransform);

	PVRTMatrixTranslation(matTransform, nPos - (fLogoSizeHalfShifted * fScaleX * nPos), -1.0f + (fLogoSizeHalfShifted * fScaleY), 0.0f);
	PVRTMatrixMultiply(matModelView, matModelView, matTransform);

	if(m_fScreenScale[0] * 640.0f<m_fScreenScale[1] * 480.0f)
	{
		PVRTMatrixRotationZ(matTransform, -90.0f*PVRT_PI/180.0f);
		PVRTMatrixMultiply(matModelView, matModelView, matTransform);
	}

	// Bind the projection and modelview matrices to the shader
	int location = glGetUniformLocation(m_pAPI->m_ProgramObject, "myMVPMatrix");
	glUniformMatrix4fv( location, 1, GL_FALSE, matModelView.f);

	// Render states
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);

	glDisable(GL_DEPTH_TEST);

	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Vertices
	glEnableVertexAttribArray(VERTEX_ARRAY);
	glEnableVertexAttribArray(UV_ARRAY);
	glEnableVertexAttribArray(COLOR_ARRAY);
	glVertexAttribPointer(VERTEX_ARRAY, 3, GL_FLOAT, GL_FALSE, 0, (const void*)pVertices);
	glVertexAttribPointer(UV_ARRAY, 2, GL_FLOAT, GL_FALSE, 0, (const void*)pUV);
	glVertexAttribPointer(COLOR_ARRAY, 4, GL_FLOAT, GL_FALSE, 0, (const void*)pColours);

	glDrawArrays(GL_TRIANGLE_STRIP,0,4);

	glDisableVertexAttribArray(VERTEX_ARRAY);
	glDisableVertexAttribArray(UV_ARRAY);
	glDisableVertexAttribArray(COLOR_ARRAY);

	// Restore render states
	glDisable (GL_BLEND);
	glEnable(GL_DEPTH_TEST);
}

/*****************************************************************************
 End of file (PVRTPrint3DAPI.cpp)
*****************************************************************************/
