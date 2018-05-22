#include "ObjectSequenceInstance.h"

#include "Utils/CoutMethods.h"

#include "glm/gtc/matrix_transform.hpp" // for glm::translate, glm::rotate, glm::scale

ObjectSequenceInstance::ObjectSequenceInstance(bool isUserMovable,
                                               int startAtFrame,
                                               const int totalFrameCount,
                                               const glm::vec4* const originalAABBCorners)
                                               : mIsUserMovable(isUserMovable), 
                                               mStartAtFrame(startAtFrame),
                                               mTotalFrameCount(totalFrameCount),
                                               mOriginalAABBCorners(originalAABBCorners)
{
   mTransformationMatrix = new GLfloat[16];

   // initialize pose to zero
   setPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
   setScaleFactor(1.0f);

   mPreviousFrameIndex = mCurrentFrameIndex;
   mBaseFrameIndex     = mStartAtFrame % mTotalFrameCount;
   mCurrentFrameIndex  = mTotalFrameCount > 1 ? mBaseFrameIndex : 0;

   mFrameStepInterval = 30.0 / 1000.0; // default value (30 ms)
   mAnimationStarted = false;
   mLooping = true;
   mPlayingForwards = true;

   mBaseViewDir = glm::vec3(0.0f, 0.0f, 1.0f);
   mViewDir = glm::vec3(0.0f, 0.0f, 1.0f);
   mViewDirRotation = 0.0f;
}

ObjectSequenceInstance::ObjectSequenceInstance(const ObjectSequenceInstance* const copy)
: mIsUserMovable(copy->mIsUserMovable), 
mStartAtFrame(copy->getCurrentFrameIndex()),
mTotalFrameCount(copy->mTotalFrameCount),
mOriginalAABBCorners(copy->mOriginalAABBCorners)
{

   mTransformationMatrix = new GLfloat[16];
   
   // initialize pose 
   mAutoRotation = copy->mAutoRotation;
   setPose(copy->mPosition.x, copy->mPosition.y, copy->mPosition.z,
      copy->mRotation.x, copy->mRotation.y, copy->mRotation.z);
   setScaleFactor(copy->mScaleFactor);

   mPreviousFrameIndex = mCurrentFrameIndex;
   mBaseFrameIndex     = mStartAtFrame % mTotalFrameCount;
   mCurrentFrameIndex  = mTotalFrameCount > 1 ? mBaseFrameIndex : 0;

   mFrameStepInterval  = copy->mFrameStepInterval ; // default value (30 ms)
   mAnimationStarted = copy->mAnimationStarted;
   mLooping = copy->mLooping;
   mPlayingForwards = copy->mPlayingForwards;

   updateTransformationMatrix();
}


ObjectSequenceInstance::~ObjectSequenceInstance()
{
   delete[] mTransformationMatrix;
}


void ObjectSequenceInstance::print()
{
   cout << "Position x: " << mPosition.x << endl;
   cout << "         y: " << mPosition.y << endl;
   cout << "         z: " << mPosition.z << endl;

   cout << "Rotation x: " << mRotation.x << endl;
   cout << "         y: " << mRotation.y << endl;
   cout << "         z: " << mRotation.z << endl;
}


void ObjectSequenceInstance::getCurrentAABB(GLfloat* boundingBox) const
{
   assert(boundingBox);
   for(int i = 0; i < 6; i++)
   {
      boundingBox[i] = mCurrentAABB[i];
   }
}

void ObjectSequenceInstance::startAnimation()
{
   mAnimationStarted = true;
   mCurrentFrameIndex = mBaseFrameIndex;
   mFrameClock.Reset();
}

void ObjectSequenceInstance::stopAnimation()
{
   mAnimationStarted = false;
   mBaseFrameIndex = mCurrentFrameIndex;
}

void ObjectSequenceInstance::setStepInterval(unsigned int ms)
{
   mFrameStepInterval  = ms / 1000.0;
}

void ObjectSequenceInstance::updateCurrentFrameIndex()
{
   if (mAnimationStarted)
   {
      double timeDelta = mFrameClock.GetElapsedTime();
      mPreviousFrameIndex = mCurrentFrameIndex;

      if(mLooping)
      {
         mCurrentFrameIndex = (mBaseFrameIndex + int(timeDelta / mFrameStepInterval )) % mTotalFrameCount;
      }
      else // play mPlayingForwards <<---->> backwards
      {
         if(mPlayingForwards)
         {
            mCurrentFrameIndex += int(timeDelta / mFrameStepInterval);
            if(mCurrentFrameIndex >= static_cast<int>(mTotalFrameCount - 1))
            {
               mCurrentFrameIndex = mTotalFrameCount - 1; // last file and now go backwards
               mPlayingForwards = false;
               //cout << "switch to backwards" << endl;
            }
         }
         else // backwards
         {
            mCurrentFrameIndex -= int(timeDelta / mFrameStepInterval);
            if(mCurrentFrameIndex <= 0)
            {
               mCurrentFrameIndex = 0; // first file and now go mPlayingForwards 
               mPlayingForwards = true;
               //cout << "switch to mPlayingForwards" << endl;
            }
         }
         if(mPreviousFrameIndex != mCurrentFrameIndex)
         {
            mFrameClock.Reset();
         }
         //cout << "mCurrentFrameIndex: " << mCurrentFrameIndex << endl;
      }

      //cout << "timeDelta         " << timeDelta << endl;
      //cout << "mBaseFrameIndex    " << mBaseFrameIndex << endl;
      //cout << "mCurrentFrameIndex "<< mCurrentFrameIndex << endl;
   }

}


bool ObjectSequenceInstance::animationStarted() const
{
   return mAnimationStarted;
}


const GLfloat* ObjectSequenceInstance::getTransformationMatrix() const
{
   return mTransformationMatrix;
}


void ObjectSequenceInstance::updateTransformationMatrix()
{
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
      glLoadIdentity();
      glTranslatef(mPosition.x, mPosition.y, mPosition.z);
      glRotatef(mRotation.x, 1, 0, 0);
      glRotatef(mRotation.y, 0, 1, 0);
      glRotatef(mRotation.z, 0, 0, 1);
      
      glRotatef(mAutoRotation.x, 1, 0, 0);
      glRotatef(mAutoRotation.y, 0, 1, 0);
      glRotatef(mAutoRotation.z, 0, 0, 1);
      
      glScalef(mScaleFactor, mScaleFactor, mScaleFactor);
      glGetFloatv(GL_MODELVIEW_MATRIX, mTransformationMatrix);
   glPopMatrix();

   updateCurrentAABB();

   // update viewing direction and view matrix
   glm::mat4 rotationMatrix; 
   rotationMatrix = glm::rotate(glm::mat4(1.0), mRotation.y, WORLD_YAXIS);
   //rotationMatrix = glm::rotate(rotationMatrix, -mViewDirRotation, WORLD_YAXIS);

   mViewDir = glm::vec3(rotationMatrix * glm::vec4(mBaseViewDir, 0.0f));

}

void ObjectSequenceInstance::updateCurrentAABB()
{
   const GLfloat* t = mTransformationMatrix; // abrrev.
   glm::mat4 tMatrix(t[0], t[1], t[2], t[3],
      t[4], t[5], t[6], t[7],
      t[8], t[9], t[10], t[11],
      t[12], t[13], t[14], t[15]);

   glm::vec4 corners[8];
   // transform original corners
   for(int i = 0 ; i < 8; i++)
   {
      corners[i] = tMatrix * mOriginalAABBCorners[i];
   }

   // compute new AABB from transformed corners
   // init with first corner
   mCurrentAABB[0] = corners[0].x;
   mCurrentAABB[1] = corners[0].y;
   mCurrentAABB[2] = corners[0].z;
   mCurrentAABB[3] = corners[0].x;
   mCurrentAABB[4] = corners[0].y;
   mCurrentAABB[5] = corners[0].z;

   for(int i = 1; i < 8; i++)
   {
      if(corners[i].x < mCurrentAABB[0]) // minX
         mCurrentAABB[0] = corners[i].x;
      if(corners[i].y < mCurrentAABB[1]) // minY
         mCurrentAABB[1] = corners[i].y;
      if(corners[i].z < mCurrentAABB[2]) // minZ
         mCurrentAABB[2] = corners[i].z;

      if(corners[i].x > mCurrentAABB[3]) // maxX
         mCurrentAABB[3] = corners[i].x;
      if(corners[i].y > mCurrentAABB[4]) // maxY
         mCurrentAABB[4] = corners[i].y;
      if(corners[i].z > mCurrentAABB[5]) // maxZ
         mCurrentAABB[5] = corners[i].z;
   }
}


void ObjectSequenceInstance::setPose(GLfloat xPos, GLfloat yPos, GLfloat zPos,
                            GLfloat xAngle, GLfloat yAngle, GLfloat zAngle)
{
   setPosition(xPos, yPos, zPos);
   setOrientation(xAngle, yAngle, zAngle);
}

void ObjectSequenceInstance::setScaleFactor(GLfloat newScale)
{
   mScaleFactor = std::max<float>(0.0001f, newScale);
   updateTransformationMatrix();
}

void ObjectSequenceInstance::setPosition(GLfloat xPos, GLfloat yPos, GLfloat zPos)
{
   setPositionX(xPos);
   setPositionY(yPos);
   setPositionZ(zPos);
   updateTransformationMatrix();
}

void ObjectSequenceInstance::setOrientation(GLfloat xAngle, GLfloat yAngle, GLfloat zAngle)
{
   setAngleX(xAngle);
   setAngleY(yAngle);
   setAngleZ(zAngle);
   updateTransformationMatrix();
}

void ObjectSequenceInstance::setAutoOrientation(GLfloat xAngle, GLfloat yAngle, GLfloat zAngle)
{
   mAutoRotation.x = xAngle;
   mAutoRotation.y = yAngle;
   mAutoRotation.z = zAngle;
   updateTransformationMatrix();
}

void ObjectSequenceInstance::move(GLfloat dx, GLfloat dy, GLfloat dz)
{
   moveX(dx);
   moveY(dy);
   moveZ(dz);
   updateTransformationMatrix();
}

void ObjectSequenceInstance::moveAlongViewingDirection(GLfloat delta)
{
   mPosition += delta * mViewDir;
   updateTransformationMatrix();
}

void ObjectSequenceInstance::rotateViewingDirection(GLfloat yAngle)
{
   mViewDirRotation += yAngle;

   glm::mat4 rotationMatrix; 
   rotationMatrix = glm::rotate(glm::mat4(1.0), -mViewDirRotation, WORLD_YAXIS);
   mBaseViewDir = glm::vec3(rotationMatrix * glm::vec4(WORLD_ZAXIS, 0.0f));

}

void ObjectSequenceInstance::rotate(GLfloat xAngle, GLfloat yAngle, GLfloat zAngle)
{
   rotateX(xAngle);
   rotateY(yAngle);
   rotateZ(zAngle);
   updateTransformationMatrix();
}

void ObjectSequenceInstance::setPositionX(GLfloat x)
{
   mPosition.x = x;
}

void ObjectSequenceInstance::setPositionY(GLfloat y)
{
   mPosition.y = y;
}

void ObjectSequenceInstance::setPositionZ(GLfloat z)
{
   mPosition.z = z;
}

void ObjectSequenceInstance::setAngleX(GLfloat x)
{
   mRotation.x = x;
}

void ObjectSequenceInstance::setAngleY(GLfloat y)
{
   mRotation.y = y;
}

void ObjectSequenceInstance::setAngleZ(GLfloat z)
{
   mRotation.z = z;
}

void ObjectSequenceInstance::moveX(GLfloat x)
{
   mPosition.x += x;
}

void ObjectSequenceInstance::moveY(GLfloat y)
{
   mPosition.y += y;
}

void ObjectSequenceInstance::moveZ(GLfloat z)
{
   mPosition.z += z;
}

void ObjectSequenceInstance::rotateX(GLfloat x)
{
   mRotation.x += x;
}

void ObjectSequenceInstance::rotateY(GLfloat y)
{
   mRotation.y += y;
}

void ObjectSequenceInstance::rotateZ(GLfloat z)
{
   mRotation.z += z;
}
