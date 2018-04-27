/******************************************************************************

 @File         TextureManager.h

 @Title        A simple texture manager for use with PVREngine

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A class for holding information about textures as they are loaded
               so that duplicate textures are not kept in memory.

******************************************************************************/
#ifndef TEXTUREMANAGER_H
#define TEXTUREMANAGER_H

/******************************************************************************
Includes
******************************************************************************/

#include "../PVRTools.h"
#include "../PVRTSingleton.h"
#include "dynamicArray.h"
#include "Globals.h"

namespace pvrengine
{

	class Texture;
	/*!****************************************************************************
	Class
	******************************************************************************/
	/*!***************************************************************************
	* @Class TextureManager
	* @Brief 	A simple texture manager for use with PVREngine.
	* @Description 	A simple texture manager for use with PVREngine.
	*****************************************************************************/
	class TextureManager : public CPVRTSingleton<TextureManager>
	{
	public:
		/*!***************************************************************************
		@Function			TextureManager
		@Description		blank constructor.
		*****************************************************************************/
		TextureManager(){}

		/*!***************************************************************************
		@Function			~TextureManager
		@Description		destructor.
		*****************************************************************************/
		~TextureManager();

		/*!***************************************************************************
		@Function			TextureManager
		@Input				i32TotalTextures	initial size of store
		@Description		blank constructor.
		*****************************************************************************/
		TextureManager(int i32TotalTextures);

		/*!***************************************************************************
		@Function			LoadTexture
		@Input				pszTextureFile	path to pvr texture to load
		@Return
		@Description		Adds a texture to the manager checking for duplicates and
		loading into texture memory.
		*****************************************************************************/
		Texture*	LoadTexture(const CPVRTString& pszTextureFile);

		/*!***************************************************************************
		@Function			GetTexture
		@Input				u32Texture	handle for texture
		@Description		Retrieves texture from manager
		*****************************************************************************/
		Texture*	GetTexture(unsigned int u32Texture) const
		{
			return m_daTextures[u32Texture];
		}

	private:
		dynamicArray<Texture*> m_daTextures;	/*! store for textures */

		/*!***************************************************************************
		@Function			Init
		@Return				success of initialisation
		@Description		common initialisation code for constructors etc.
		*****************************************************************************/
		bool Init();

	};

}
#endif // TEXTUREMANAGER_H

/******************************************************************************
End of file (TextureManager.h)
******************************************************************************/
