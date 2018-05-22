#ifndef OBJECTSEQUENCE_H
#define OBJECTSEQUENCE_H

#include <vector>
#include <ctime>

#include "OpenGL.h"

#include "glm/glm.hpp"

using namespace std;

class ObjModel;
class ObjectSequenceInstance;

/// An ObjectSequence consists of several Wavefront OBJ Models. 
/// The model sequence is a predefined animation displaying one rigid body after the other.
/// A sequence may also contain only one model (= no animation).
/// It has at least one instance.
class ObjectSequence
{
public:
   ObjectSequence(string name,
      int atlasWidth, int atlasHeight,
      ObjModel* firstModel,
      bool dynamic,
      string pathSequence ="", int sequenceReadInMethod = -1
      );
   ~ObjectSequence();

   /// Adds an ObjModel to the sequence
   void addObjModel(ObjModel* model, int fileIndex);
   void addInstance(bool isUserMovable, int startAtFrame);
   void addInstance(const ObjectSequenceInstance* const copy);
   
   /// Stop or start the animation
   void startAnimation(int instanceIndex);
   void stopAnimation(int instanceIndex);

   void updateCurrentFrameIndex(int instanceIndex);

   /// Calculates the axis-aligned boundingBox 
   /// (minX, minY, minZ, maxX, maxY, maxZ) 
   /// of the models in this sequence.
   /// Must be called after all models have been added to this sequence.
   void computeOriginalAABB(); 
   void updateAllCurrentAABBs();

   /// Sets a new atlas resolution
   void setAtlasWidth(int newWidth) { mAtlasWidth = newWidth; }
   void setAtlasHeight(int newHeight) { mAtlasHeight = newHeight; }

   /// \param mode  - a bitwise OR of values describing what is to be rendered.
   ///             GLM_NONE     -  render with only vertices
   ///             GLM_FLAT     -  render with facet normals
   ///             GLM_SMOOTH   -  render with vertex normals
   ///             GLM_TEXTURE  -  render with texture coords
   ///             GLM_COLOR    -  render with colors (color material)
   ///             GLM_MATERIAL -  render with materials
   ///             GLM_COLOR and GLM_MATERIAL should not both be specified.  
   ///             GLM_FLAT and GLM_SMOOTH should not both be specified.  
   void setDefaultDrawMode(GLuint mode);

   // GETTER

   ObjectSequenceInstance* getInstance(int instanceIndex) const { return mInstances.at(instanceIndex); }
   unsigned int getNumInstances() const { return mInstances.size(); }
   bool hasMovableInstances() const { return mHasMovableInstances; } 
      
   /// Only for static elements.
   void getCurrentAABB(GLfloat* boundingBox)  const;

   int getLoadedModelCount() const { return mObjSequence.size(); }
   int getAnimFileStartIndex() const { return mAnimFileStartIndex; }
   int getAnimFileEndIndex() const { return mAnimFileEndIndex; }

   /// Returns the pointer of the current object model in the sequence.
   ObjModel* getCurrentObjModel(int instanceIndex) const;

   /// Returns whether this element is static (not animated, not circling, not user-movable)
   bool isStatic() const { return !mDynamic; }

   string getName() const { return mName; }
   string getPathSequence() const { return mPathSequence; }
   int getSequenceReadInMethod() const { return mSequenceReadInMethod; }
   int getAtlasWidth() const { return mAtlasWidth; }
   int getAtlasHeight() const { return mAtlasHeight; }

   /// \param[out] boundingBox - array of 6 GLfloats (GLfloat boundingBox[6])
   void getOriginalAABB(GLfloat* boundingBox) const;

   /// Direct access to models in this sequence.
   ObjModel* getModel(int index) const { return mObjSequence.at(index); }

   /// Returns the number of triangles of the first model
   /// (assumption is that all models in the sequence have the same number of triangles)
   int getTriangleCount() const;

   GLuint getDefaultDrawMode() const;

private:

   ObjectSequence();

   // Member data

   GLuint mDefaultDrawMode;
   GLfloat mOriginalAABB[6]; ///< minX, minY, minZ, maxX, maxY, maxZ (after initial model loading)
   glm::vec4* mOriginalAABBCorners;

   //@{
   /// Resolution of texture atlas for this object
   int mAtlasWidth;  
   int mAtlasHeight; 
   //@}

   // Obj file reading
   const string mName;
   const string mPathSequence; 
   const int mSequenceReadInMethod;

   const bool mDynamic; ///< is this object sequence static or dynamic?
   bool mHasMovableInstances; ///< model instances may be moved by the user

   vector<ObjModel*> mObjSequence; ///< holds pointer to all (properly loaded and initialized) ObjModels
   vector<ObjectSequenceInstance*> mInstances; ///< must have at least one element

   int mAnimFileStartIndex, mAnimFileEndIndex;


};

#endif
