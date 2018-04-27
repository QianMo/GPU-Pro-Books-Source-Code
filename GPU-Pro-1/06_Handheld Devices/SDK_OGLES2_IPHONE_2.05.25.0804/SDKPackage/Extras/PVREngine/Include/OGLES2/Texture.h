/******************************************************************************

 @File         Texture.h

 @Title        A class to hold runtime info about textures in order to shield the

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A class for holding information about textures as they are loaded.

******************************************************************************/
#ifndef TEXTURE_H
#define TEXTURE_H

#include "../PVRTools.h"

namespace pvrengine
{
	/*!***************************************************************************
	* @Class Texture
	* @Brief 	A class for holding information about textures as they are loaded.
	* @Description 	A class for holding information about textures as they are loaded.
	*****************************************************************************/
	class Texture
	{
	public:
		/*!***************************************************************************
		@Function			Texture
		@Description		blank constructor
		*****************************************************************************/
		Texture(){}
		/*!***************************************************************************
		@Function			Texture
		@Input				strFilename
		@Description		Constructor from .pvr file using the PVRTools loading code
		*****************************************************************************/
		Texture(const CPVRTString& strFilename);
		/*!***************************************************************************
		@Function			getHandle
		@Description		accessor for OGL handle
		*****************************************************************************/
		unsigned int getHandle() {return m_u32Handle;}
		/*!***************************************************************************
		@Function			getFilename
		@Description		accessor for source texture file name
		*****************************************************************************/
		CPVRTString getFilename(){return m_strFilename;}
		/*!***************************************************************************
		@Function			isCubeMap
		@Return				true for cube map
		@Description		Function to query whether this is a cube map texture
		*****************************************************************************/
		bool	isCubeMap()	{return m_bIsCubeMap;}

	private:
		unsigned int m_u32Handle;		/*! GL identifier for this texture */
		CPVRTString m_strFilename;		/*! path to this texture's original file */
		bool m_bIsCubeMap;				/*! stores whether texture is a cube map */
	};
}
#endif // TEXTURE_H

/******************************************************************************
End of file (TextureBase.h)
******************************************************************************/
