#include "TexturePool.h"


map<string, GLuint> TexturePool::mTextureIDs;

GLuint TexturePool::getTexture(string name)
{
	GLuint textureID = 0;

	map<string, GLuint>::iterator textureIDsIterator;
	textureIDsIterator = mTextureIDs.find(name);

	if(textureIDsIterator != mTextureIDs.end())
   {
		textureID = textureIDsIterator->second;	
	}
   else
   {
      cout << "[WARNING] TexturePool::getTexture("<<name<<") : Did not find texture with given name. Return 0." << endl;
   }
   return textureID;

}

bool TexturePool::addTexture(string name, GLuint textureID)
{
   map<string, GLuint>::iterator textureIDsIterator = mTextureIDs.find(name);

   // found name
   if(textureIDsIterator != mTextureIDs.end())
   {
      GLuint id = textureIDsIterator->second;
      if(id == textureID)
      {
         cout << "TexturePool::addTexture("<<name<<","<<textureID<<") : NOTE: There already exists a texture with the same name and id." << endl;
         return true;
      }
      else
      {
         cout << "TexturePool::addTexture("<<name<<","<<textureID<<") : CONFLICT: There exists a texture with the same name, but a different id." << endl;
         return false;
      }
   }
   // name not found: insert
   else
   {
      std::cout << "TexturePool::addTexture("<<name<<","<<textureID<<")" << std::endl;

      mTextureIDs[name] = textureID;	
      return true;
   }
   
}

void TexturePool::deleteTexture(string name)
{
   map<string, GLuint>::iterator it = mTextureIDs.find(name);
   if(it != mTextureIDs.end())
   {
      if(glIsTexture(it->second))
         glDeleteTextures(1, &(it->second));
      mTextureIDs.erase(it);   // erasing by iterator
   }
   else // name not found, nothing to do
   {
      cout << "TexturePool::deleteTexture("<<name<<") : NOTE: A texture with this name does not exit." << endl;
   }
}

