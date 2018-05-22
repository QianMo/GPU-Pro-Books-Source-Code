///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#ifndef ATLASRENDERER_H
#define ATLASRENDERER_H

#include "OpenGL.h"

#include <vector>
using std::vector;

struct PixelCoordinate
{
   int x, y;
};

class ShaderProgram;

/// AtlasRenderer holds one position atlas with world coordinates
/// and a display list with vertices for point rendering 
/// for each scene element (= ObjectSequence) 
class AtlasRenderer
{

public:
   AtlasRenderer();

   void updateAllTextureAtlases();

   void changeAtlasResolutionRelative(int elementIndex, unsigned int deltaX, unsigned int deltaY);
   
   /// Returns the position texture atlas
   GLuint getTextureAtlas(int elementIndex, int instanceIndex) { return mAtlases.at(elementIndex).at(instanceIndex); }
   const vector<vector<GLuint> >& getAtlases() const { return mAtlases; }

   GLuint getPixelDisplayList(int elementIndex) { return mPixelDisplayLists.at(elementIndex); }

   /// Creates the textures for the atlas of scene element with index elementIndex.
   void createAtlas(int elementIndex, bool addNew);

private:
   void initialAtlasRendering();

   /// Render texture atlas with 3D positions for all instances of the given scene element
   void renderTextureAtlas(int elementIndex);

   /// Change the atlas resolution for the given element 
   void changeAtlasResolution(int elementIndex, unsigned int newWidth, unsigned int newHeight);

   void createFBO();
   void createShader();

   void createPixelDisplayList(int elementIndex);

   // Texture Atlas
   GLuint mFBOAtlas;
   vector<vector<GLuint> > mAtlases; ///< RGB: position
   // one atlas for every instance 
   // mAtlases.at(e) => sceneElementAtlases
   // mAtlases.at(e).at(i) => instance i (texture handle of the atlas instance i of sceneElement e )

   ShaderProgram* pAtlasPosition; ///< outputs world positions (x, y, z) to a RGB Texture

   vector<GLuint> mPixelDisplayLists; // display lists (one for each scene element) with vertices for valid atlas texels

};


#endif
