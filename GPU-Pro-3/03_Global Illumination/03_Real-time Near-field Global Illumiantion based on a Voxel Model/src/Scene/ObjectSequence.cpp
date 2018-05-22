#include "ObjectSequence.h"

#include "ObjModel.h"
#include "ObjectSequenceInstance.h"

ObjectSequence::ObjectSequence(string name, 
                               int atlasWidth, int atlasHeight,
                               ObjModel* firstModel,
                               bool dynamic,
                               string pathSequence, int sequenceReadInMethod)                              
: mDynamic(dynamic), 
  mName(name),
  mPathSequence(pathSequence),
  mSequenceReadInMethod(sequenceReadInMethod),
  mAtlasWidth(atlasWidth), mAtlasHeight(atlasHeight)
{
   mOriginalAABBCorners = new glm::vec4[8]; // Box has eight corners

   mDefaultDrawMode = GLM_NONE;
   mHasMovableInstances = false;

   // animation sequence
   if(firstModel) addObjModel(firstModel, 0); 
   mAnimFileStartIndex = 0;
   mAnimFileEndIndex = 0;

   if(!mDynamic) // static elements only have one instance
   {
      addInstance(false, 0);
   }
}

ObjectSequence::~ObjectSequence()
{
   for(unsigned int i = 0; i < mObjSequence.size(); i++)
   {
      delete mObjSequence.at(i);
   }
}



void ObjectSequence::getOriginalAABB(GLfloat* boundingBox) const
{
   assert(boundingBox);
   for(int i = 0; i < 6; i++)
   {
      boundingBox[i] = mOriginalAABB[i];
   }
}

void ObjectSequence::getCurrentAABB(GLfloat *boundingBox) const
{
   //if(!dynamic)
   {
      mInstances.front()->getCurrentAABB(boundingBox);
   }
}

void ObjectSequence::updateAllCurrentAABBs()
{
   for(unsigned int i = 0; i < mInstances.size(); i++)
   {
      mInstances.at(i)->updateTransformationMatrix();
      mInstances.at(i)->updateCurrentAABB();
   }
}


void ObjectSequence::computeOriginalAABB()
{
   GLfloat bb[6];
   mObjSequence.at(0)->glmBoundingBox(bb);
   GLfloat minX = bb[0];
   GLfloat minY = bb[1];
   GLfloat minZ = bb[2];
   GLfloat maxX = bb[3];
   GLfloat maxY = bb[4];
   GLfloat maxZ = bb[5];

   // compute bounding box for every model
   for(unsigned int i = 1; i < mObjSequence.size(); i++)
   {
      mObjSequence.at(i)->glmBoundingBox(bb);
      if(bb[0] < minX) minX = bb[0];
      if(bb[1] < minY) minY = bb[1];
      if(bb[2] < minZ) minZ = bb[2];
      if(bb[3] > maxX) maxX = bb[3];
      if(bb[4] > maxY) maxY = bb[4];
      if(bb[5] > maxZ) maxZ = bb[5];
   }

   bb[0] = minX;
   bb[1] = minY;
   bb[2] = minZ;
   bb[3] = maxX;
   bb[4] = maxY;
   bb[5] = maxZ;

   for(int i = 0; i < 6; i++)
   {
      mOriginalAABB[i] = bb[i];
   }


   // Compute and save the corners of the original AABB 
   for(int i = 0; i < 8; i++)
   {
      mOriginalAABBCorners[i] = glm::vec4(
         (((i & 1) == 0) ? mOriginalAABB[0] : mOriginalAABB[3]),
         (((i & 2) == 0) ? mOriginalAABB[1] : mOriginalAABB[4]),
         (((i & 4) == 0) ? mOriginalAABB[2] : mOriginalAABB[5]),
         1.0f);
   }

}

int ObjectSequence::getTriangleCount() const
{ return mObjSequence.front()->getTriangleCount(); }

void ObjectSequence::addObjModel(ObjModel* model, int fileIndex)
{
   mObjSequence.push_back(model);
   if(mObjSequence.size() == 1)
   {
      // first model added
      mAnimFileStartIndex = fileIndex;
   }
   // (last added model => last file index of animation)
   mAnimFileEndIndex = fileIndex;
}

void ObjectSequence::addInstance(bool isUserMovable, int startAtFrame)
{
   // static elements only have one instance
   if(mInstances.empty() || mDynamic)
   {
      mInstances.push_back(new ObjectSequenceInstance(
         isUserMovable,
         startAtFrame,
         mObjSequence.size(),
         mOriginalAABBCorners));
      mHasMovableInstances |= isUserMovable;
   }
}

void ObjectSequence::addInstance(const ObjectSequenceInstance* const copy)
{
   if(mDynamic)
   {
      mInstances.push_back(new ObjectSequenceInstance(copy));
   }
}



void ObjectSequence::setDefaultDrawMode(GLuint mode)
{
   mDefaultDrawMode = mode;
}

GLuint ObjectSequence::getDefaultDrawMode() const
{
   return mDefaultDrawMode;
}

void ObjectSequence::startAnimation(int instanceIndex)
{
   if(mObjSequence.size() > 1)
   {
      mInstances.at(instanceIndex)->startAnimation();
   }
}

void ObjectSequence::stopAnimation(int instanceIndex)
{
   if(mObjSequence.size() > 1)
   {
      mInstances.at(instanceIndex)->stopAnimation();
   }
}



void ObjectSequence::updateCurrentFrameIndex(int instanceIndex)
{
   if (mObjSequence.size() > 1 )
   {
      mInstances.at(instanceIndex)->updateCurrentFrameIndex();
   }
}

ObjModel* ObjectSequence::getCurrentObjModel(int instanceIndex) const
{
   return mObjSequence.at(mInstances.at(instanceIndex)->getCurrentFrameIndex());
}

