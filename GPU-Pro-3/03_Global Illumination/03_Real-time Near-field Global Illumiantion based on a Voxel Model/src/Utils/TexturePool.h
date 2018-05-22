#ifndef TEXTUREPOOL_H
#define TEXTUREPOOL_H

#include "OpenGL.h"

#include <map>
#include <string>
#include <iostream>

using namespace std;

/// Manages common used textures such as the voxelization bitmasks.
class TexturePool
{
public:

   /// Returns the texture id of texture with given name.
   /// Returns 0 if texture with given name does not exist.
   static GLuint getTexture(string name);

   /// Adds an OpenGL Texture ID associated to given name to the map textureIDs.
   /// Checks whether a texture with this name already exits.
   /// Returns if successful.
   static bool addTexture(string name, GLuint textureID);

   /// Deletes the texture and removes the pair (name, id) from the map.
   static void deleteTexture(string name);


private:

   static map<string, GLuint> mTextureIDs; ///< maps texture names to OpenGL texture handle IDs
};

#endif
