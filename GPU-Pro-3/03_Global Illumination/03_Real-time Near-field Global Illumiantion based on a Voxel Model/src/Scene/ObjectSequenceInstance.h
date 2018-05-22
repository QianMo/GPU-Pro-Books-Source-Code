#ifndef OBJECTSEQUENCEINSTANCE_H
#define OBJECTSEQUENCEINSTANCE_H

#include "SFML/System/Clock.hpp"

#include "OpenGL.h"
#include "glm/glm.hpp"

#include <iostream>

using namespace std;

class ObjectSequenceInstance 
{
public:
   ObjectSequenceInstance(bool isUserMovable,
      int startAtFrame, const int totalFrameCount,
      const glm::vec4* const originalAABBCorners);

   ObjectSequenceInstance(const ObjectSequenceInstance* const copy);

   ~ObjectSequenceInstance();

   void updateCurrentFrameIndex();

   /// Bounding box computation

   /// Calculates the AABB of this object's transformed bounding box.
   /// Rotates and translates the original AABB and computes a new
   /// AABB containing this rotated BB.
   void updateCurrentAABB();

   /// Prints pose information.
   void print();

   /// SETTER //////////////////////////////

   /// Starts index calculation (which model in the sequence is the current model)
   /// for the object sequence animation - only if this object is tagged as dynamic!
   void startAnimation();
   /// Stop index calculation, current index is the last computed sequence index.
   void stopAnimation();


   /// Sets the time after which the next model in the sequence will be the current model.
   void setStepInterval(unsigned int ms);
   void setLooping(bool looping) { mLooping = looping; }
   void setForwards(bool forwards) { mPlayingForwards = forwards; }


   /// Set object's pose (position and orientation).
   void setPose(GLfloat xPos, GLfloat yPos, GLfloat zPos,
                GLfloat xAngle, GLfloat yAngle, GLfloat zAngle);

   void setScaleFactor(GLfloat newScale);


   //@{
   /// Set/modify object's pose.
   void setPosition(GLfloat xPos, GLfloat yPos, GLfloat zPos);
   void setOrientation(GLfloat xAngle, GLfloat yAngle, GLfloat zAngle);
   void setAutoOrientation(GLfloat xAngle, GLfloat yAngle, GLfloat zAngle);
   void move(GLfloat dx, GLfloat dy, GLfloat dz);
   void moveAlongViewingDirection(GLfloat delta);
   void rotateViewingDirection(GLfloat yAngle);
   void rotate(GLfloat xAngle, GLfloat yAngle, GLfloat zAngle);
   //@}


   /// GETTER ///////////////////////////
   /// Return the object's transformation matrix (16 elements, column-major).
   const GLfloat* getTransformationMatrix() const;
   const GLfloat* getPreviousTransformationMatrix() const;
   
   /// Returns the current AABB of this (transformed) object.
   const GLfloat* getCurrentAABB()  const { return mCurrentAABB; }
   /// For convenience. Returns AABB of first instance.
   /// \param[out] boundingBox - array of 6 GLfloats (GLfloat boundingBox[6])
   void getCurrentAABB(GLfloat* boundingBox)  const;

   int getCurrentFrameIndex() const { return mCurrentFrameIndex; }
   bool isUserMovable() const { return mIsUserMovable; }

   /// Returns whether the animation has been started;
   bool animationStarted() const;
   bool isLooping() const { return mLooping; }
   bool playingForwards() const { return mPlayingForwards; }
   unsigned int getStepInterval() const { return static_cast<unsigned int>(mFrameStepInterval * 1000.0); }

   const glm::vec3& getPosition() const { return mPosition; }
   const glm::vec3& getRotation() const { return mRotation; }
   float getScaleFactor() const { return mScaleFactor; }

   const glm::vec3& getViewDir() const { return mViewDir; }

   void updateTransformationMatrix();

private:
   ObjectSequenceInstance();

   // Member methods

   void setPositionX(GLfloat x);
   void setPositionY(GLfloat y);
   void setPositionZ(GLfloat z);

   void setAngleX(GLfloat x);
   void setAngleY(GLfloat y);
   void setAngleZ(GLfloat z);

   void moveX(GLfloat x);
   void moveY(GLfloat y);
   void moveZ(GLfloat z);

   void rotateX(GLfloat angle);
   void rotateY(GLfloat angle);
   void rotateZ(GLfloat angle);

   // Member data

   const bool mIsUserMovable; // May the user move this object?

   GLfloat mCurrentAABB[6];  ///< minX, minY, minZ, maxX, maxY, maxZ in world space
   const glm::vec4* const mOriginalAABBCorners;

   glm::vec3 mPosition;   ///< World space position
   glm::vec3 mRotation;   ///< Orientation (in degrees: xAngle, yAngle, zAngle)
   glm::vec3 mAutoRotation; ///< automatical rotation around a world space axis
   glm::vec3 mBaseViewDir; 
   glm::vec3 mViewDir; ///< viewing vector of this object instance
   float mViewDirRotation; 
   GLfloat mScaleFactor;

   GLfloat* mTransformationMatrix; ///< column-major OpenGL transformation matrix

   // (obj-sequence) Animation

   int mStartAtFrame;    ///< animation starts at this frame number
   const int mTotalFrameCount; ///< total number of frames this animation sequence has
   double mFrameStepInterval; ///< time (in s) to be waited for, before the next model in the sequence will become the current model

   sf::Clock mFrameClock; ///< The clock starts automatically after being constructed
   
   int mBaseFrameIndex; ///< offset to the start
   int mCurrentFrameIndex; ///< offset relative to baseIndex 
   int mPreviousFrameIndex; 

   bool mAnimationStarted; ///< indicates whether animation has been started
   bool mLooping; ///< if true the animation will be looping; else it will go forwards and backwards
   bool mPlayingForwards; ///< if not looping, this boolean indicates whether we are running the animation for- or backwards 
};


#endif
