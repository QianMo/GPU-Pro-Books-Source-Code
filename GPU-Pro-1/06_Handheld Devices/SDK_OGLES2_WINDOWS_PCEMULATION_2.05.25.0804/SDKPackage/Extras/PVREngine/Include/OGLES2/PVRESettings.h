/******************************************************************************

 @File         PVRESettings.h

 @Title        PVREngine main header file for OGLES2 API

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent/OGLES2

 @Description  API independent settings for the PVREngine

******************************************************************************/
#ifndef PVRESETTINGS_H
#define PVRESETTINGS_H

#include "../PVRTools.h"

namespace pvrengine
{

	/*!  constants to aid in API abstraction */

	/*!  clear constants */
	const int PVR_COLOUR_BUFFER		= GL_COLOR_BUFFER_BIT;
	const int PVR_DEPTH_BUFFER		= GL_DEPTH_BUFFER_BIT;

	/*!  general constants */
	const int PVR_NONE				= GL_NONE;
	const int PVR_FRONT				= GL_FRONT;
	const int PVR_BACK				= GL_BACK;
	const int PVR_FRONT_AND_BACK	= GL_FRONT_AND_BACK;

	/*!***************************************************************************
	* @Class PVRESettings
	* @Brief 	API independent settings for the PVREngine.
	* @Description 	API independent settings for the PVREngine.
	*****************************************************************************/
	class PVRESettings
	{
	public:
		/*!***************************************************************************
		@Function			PVRESettings
		@Description		blank constructor.
		*****************************************************************************/
		PVRESettings();

		/*!***************************************************************************
		@Function			~PVRESettings
		@Description		destructor.
		*****************************************************************************/
		~PVRESettings();

		/*!***************************************************************************
		@Function			PVRESettings
		@Description		API initialisation code.
		*****************************************************************************/
		void Init();

		/*!***************************************************************************
		@Function			setBackColour
		@Input				cBackColour - clear colour
		@Description		sets the clear colour
		*****************************************************************************/
		void setBackColour(const unsigned int cBackColour);

		/*!***************************************************************************
		@Function			setBackColour
		@Input				fRed - red component of colour
		@Input				fGreen - red component of colour
		@Input				fBlue - red component of colour
		@Input				fAlpha - red component of colour
		@Description		sets the clear colour
		*****************************************************************************/
		void setBackColour(const float fRed, const float fGreen, const float fBlue, const float fAlpha);

		/*!***************************************************************************
		@Function			setBackColour
		@Input				fRed - red component of colour
		@Input				fGreen - red component of colour
		@Input				fBlue - red component of colour
		@Description		sets the clear colour
		*****************************************************************************/
		void setBackColour(const float fRed, const float fGreen, const float fBlue);

		/*!***************************************************************************
		@Function			setBackColour
		@Input				u32Red - red component of colour
		@Input				u32Green - red component of colour
		@Input				u32Blue - red component of colour
		@Input				u32Alpha - red component of colour
		@Description		sets the clear colour
		*****************************************************************************/
		void setBackColour(const unsigned int u32Red,
			const unsigned int u32Green,
			const unsigned int u32Blue,
			const unsigned int u32Alpha);

		/*!***************************************************************************
		@Function			setBackColour
		@Input				u32Red - red component of colour
		@Input				u32Green - red component of colour
		@Input				u32Blue - red component of colour
		@Description		sets the clear colour
		*****************************************************************************/
		void setBackColour(const unsigned int u32Red,
			const unsigned int u32Green,
			const unsigned int u32Blue);

		/*!***************************************************************************
		@Function			setClearFlags
		@Input				u32ClearFlags - see constants above
		@Description		sets which buffers that the clear operation affects
		*****************************************************************************/
		void setClearFlags(unsigned int u32ClearFlags);

		/*!***************************************************************************
		@Function			getClearFlags
		@Return				the current clear settnigs - see constants above
		@Description		sets which buffers that the clear operation affects
		*****************************************************************************/
		unsigned int getClearFlags();

		/*!***************************************************************************
		@Function			setDepthTest
		@Input				bDepth - true test, false don't
		@Description		switches the z depth test on or off
		*****************************************************************************/
		void setDepthTest(const bool bDepth);

		/*!***************************************************************************
		@Function			setDepthWrite
		@Input				bDepthWrite - true write, false don't
		@Description		sets whether rendering writes to the z depth buffer
		*****************************************************************************/
		void setDepthWrite(const bool bDepthWrite);

		/*!***************************************************************************
		@Function			setBlend
		@Input				bBlend - true blend, false don't
		@Description		sets whether alpha blending is enabled according to the
		current blend mode
		*****************************************************************************/
		void setBlend(const bool bBlend);

		/*!***************************************************************************
		@Function			setCull
		@Input				bCull - true cull, false don't
		@Description		sets whether culling is enabled according to the current
		cull mode
		*****************************************************************************/
		void setCull(const bool bCull);

		/*!***************************************************************************
		@Function			setCullMode
		@Input				eMode - the cull mode
		@Description		sets the cull mode
		*****************************************************************************/
		void setCullMode(unsigned int eMode);

		/*!***************************************************************************
		@Function			Clear
		@Description		Performs a clear
		*****************************************************************************/
		void Clear();

		/*!***************************************************************************
		@Function			setClearFlags
		@Input				sPrint3d - reference to the Print3D class for this context
		@Input				u32Width - width of viewport
		@Input				u32Height - height of viewport
		@Input				bRotate - rotate for portrait flag
		@Description		Mandatory
		*****************************************************************************/
		bool InitPrint3D(CPVRTPrint3D& sPrint3d,
			const unsigned int u32Width,
			const unsigned int u32Height,
			const bool bRotate);

		/*!***************************************************************************
		@Function			getAPIName
		@Return				human readable string of the current API
		@Description		Returns a string containing the name of the current API:
		e.g. "OpenGL"
		*****************************************************************************/
		CPVRTString getAPIName();

	protected:
		unsigned int		m_u32ClearFlags;	/*! current value to clear with */
	};

}
#endif // PVRESETTINGS_H

/******************************************************************************
End of file (PVRESettings.h)
******************************************************************************/
