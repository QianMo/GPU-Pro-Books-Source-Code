/******************************************************************************

 @File         TextureManager.cpp

 @Title        Introducing the POD 3d file format

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Shows how to use the pfx format

******************************************************************************/
#include "TextureManager.h"
#include "Texture.h"
#include "ConsoleLog.h"

namespace pvrengine
{
	

	/******************************************************************************/

	TextureManager::~TextureManager()
	{
		for(unsigned int i=0;i<m_daTextures.getSize();++i)
		{	// check if Texture is already in manager
			for(unsigned int j=i+1;j<m_daTextures.getSize();++j)
			{
				if(m_daTextures[i]==m_daTextures[j])
				{
					m_daTextures[i]=NULL;
					break;
				}
			}
		}

		for(unsigned int i=0;i<m_daTextures.getSize();++i)
		{
			PVRDELETE(m_daTextures[i]);
		}
	}

	/******************************************************************************/

	TextureManager::TextureManager(int i32MaxTextures)
	{
		m_daTextures.expandToSize(i32MaxTextures);
	}

	/******************************************************************************/

	Texture* TextureManager::LoadTexture(const CPVRTString& strTextureFile)
	{
		unsigned int u32NumTextures = m_daTextures.getSize();
		// check if this texture has already been loaded
		for(unsigned int i=0;i<u32NumTextures;i++)
		{
			if(m_daTextures[i]->getFilename().compare(strTextureFile)==0)
			{	// file already present so retrieve the texture info and return
				ConsoleLog::inst().log("Texture %s already in manager.\n", m_daTextures[i]->getFilename().c_str());
				m_daTextures.append(m_daTextures[i]);
				return m_daTextures[i];
			}
		}

		// store filename, handle and header
		m_daTextures.append(new Texture(strTextureFile));
		if(!m_daTextures[u32NumTextures])
		{
			ConsoleLog::inst().log("Loading texture failed.");
		}

		return m_daTextures[u32NumTextures];
	}

}

/******************************************************************************
End of file (TextureManager.cpp)
******************************************************************************/
