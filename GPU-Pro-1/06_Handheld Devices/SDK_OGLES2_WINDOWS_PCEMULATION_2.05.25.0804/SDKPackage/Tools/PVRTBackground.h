/******************************************************************************

 @File         PVRTBackground.h

 @Title        PVRTBackground

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Function to draw a background texture.

******************************************************************************/
#ifndef __PVRTBACKGROUND_H__
#define __PVRTBACKGROUND_H__

#include "PVRTGlobal.h"
#include "PVRTContext.h"
#include "PVRTString.h"
#include "PVRTError.h"

/****************************************************************************
** Structures
****************************************************************************/
/*!***************************************************************************
@Struct SPVRTBackgroundAPI
@Brief A struct for storing API specific variables
*****************************************************************************/
struct SPVRTBackgroundAPI;

/*!***************************************************************************
@Class CPVRTBackground
@Brief A class for drawing a fullscreen textured background
*****************************************************************************/
class CPVRTBackground
{
	public:
		/*!***************************************************************************
		 @Function			CPVRTBackground
	 	 @Description		Init some values.
		*****************************************************************************/
		CPVRTBackground(void);
		/*!***************************************************************************
		 @Function			~CPVRTBackground
		 @Description		Calls Destroy()
		*****************************************************************************/
		~CPVRTBackground(void);
		/*!***************************************************************************
		 @Function		Destroy
		 @Description	Destroys the background. It's called by the destructor.
		*****************************************************************************/
		void Destroy();
		/*!***************************************************************************
		 @Function		Init
		 @Input			pContext	A pointer to a PVRTContext
		 @Input			bRotate		true to rotate texture 90 degrees.
		 @Input			pszError	An option string for returning errors
		 @Return 		PVR_SUCCESS on success
		 @Description	Initialises the background
		*****************************************************************************/
		EPVRTError Init(const SPVRTContext * const pContext, const bool bRotate, CPVRTString *pszError = 0);

#if defined(BUILD_OGL) || defined(BUILD_OGLES) || defined(BUILD_OGLES2)
		/*!***************************************************************************
		 @Function		Draw
		 @Input			ui32Texture	Texture to use
		 @Return 		PVR_SUCCESS on success
		 @Description	Draws a texture on a quad covering the whole screen.
		*****************************************************************************/
		EPVRTError Draw(const GLuint ui32Texture);
#elif defined(BUILD_DX10)
		/*!***************************************************************************
		 @Function		Draw
		 @Input			pTexture	Texture to use
		 @Return 		PVR_SUCCESS on success
		 @Description	Draws a texture on a quad covering the whole screen.
		*****************************************************************************/
		EPVRTError Draw(ID3D10ShaderResourceView *pTexture);
#endif

	protected:
		bool m_bInit;
		SPVRTBackgroundAPI *m_pAPI;
};


#endif /* __PVRTBACKGROUND_H__ */

/*****************************************************************************
 End of file (PVRTBackground.h)
*****************************************************************************/
