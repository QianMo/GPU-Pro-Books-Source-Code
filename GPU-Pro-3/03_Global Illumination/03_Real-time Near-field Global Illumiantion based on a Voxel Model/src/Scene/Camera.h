#ifndef CAMERA_H
#define CAMERA_H

#include "OpenGL.h"

/// Structure that defines the camera frustum parameters.
struct Frustum
{
   GLfloat left;
   GLfloat right;
   GLfloat bottom;
   GLfloat top;
   GLfloat zNear;
   GLfloat zFar;
   GLfloat width;  // right - left
   GLfloat height; // top   - bottom
   GLfloat zRange; // zFar  - zNear
};


#include "glm/glm.hpp" // OpenGL Mathematics Library

#include <iostream>
using namespace std;

/// A simple camera for a ego perspective

class Camera
{
public:
   Camera();
   ~Camera();


   //------- Frustum / Projection --------//

   /// Computes perspective frustum parameters
   /// and sets perspective projection matrix.
   void setPerspectiveFrustum(GLfloat fovh, GLfloat aspect, GLfloat zNear, GLfloat zFar);

   /// Sets orthographic frustum parameters
   /// and sets orthographic projection matrix.
   void setOrthographicFrustum(GLfloat left, GLfloat right, GLfloat bottom, GLfloat top, GLfloat zNear, GLfloat zFar);

   /// Sets the horizontal field of view
   void setFovH(float newFovH);

   /// Sets the near plane
   void setZNear(float newZNear);

   /// Matrices

   const glm::mat4& getViewMatrix() const
   { return mViewMatrix; }
   const glm::mat4& getInverseViewMatrix() const
   { return mInverseViewMatrix; }
   const glm::mat4& getViewProjectionMatrix() const // projection * view
   { return mViewProjMatrix; }
   const GLfloat* getProjectionMatrix() const
   { return &mProjectionMatrix[0][0]; }
   const glm::mat4& getViewProjectionToUnitMatrix() const
   { return mViewProjectionToUnitMatrix; } 
   const glm::mat4& getInverseViewProjectionToUnitMatrix() const
   { return mInverseViewProjectionToUnitMatrix; } 

   const Frustum& getFrustum() const { return mFrustum; }

   float getFovh() const   { return mFovh; }
   float getAspect() const { return mAspect; }


   //------ Viewing Transformation -------/

   void setUserPosition(float x, float y, float z);
   void setUserPosition(glm::vec3 newP);

   /// Set angle for rotating around x-axis in degrees (pitch).
   void setAngleX(GLfloat angle);
   /// Set angle for rotating around y-axis in degrees (yaw). 
   void setAngleY(GLfloat angle);
   /// Set angle for rotating around z-axis in degrees (roll).
   void setAngleZ(GLfloat angle);

   float getAngleX() const { return mAngleX; }
   float getAngleY() const { return mAngleY; }
   float getAngleZ() const { return mAngleZ; }

   /// Modifies value by adding delta.
   //@{
   void modifyAngleX(GLfloat delta);
   void modifyAngleY(GLfloat delta);
   void modifyAngleZ(GLfloat delta);
   void move(GLfloat deltaX, GLfloat deltaY, GLfloat deltaZ); 
   //@}

   static void setWindowSize(int windowWidth, int windowHeight);

   /// Simulates two cameras whose viewing directions are
   /// orthogonal to this camera's viewing direction.
   /// The orthogonal cameras orbit around this camera's frustum center.
   /// Frustum must be cubic for matching.
   /// Calling this function enables computation for the two
   /// orthogonal viewing matrices.
   void enableOrthogonalViews();

   const glm::mat4& getViewMatrixOrthogonalX() const
      { return mViewMatrixOrthogonalX; }
   const glm::mat4& getViewMatrixOrthogonalY() const
      { return mViewMatrixOrthogonalY; }
   const glm::mat4& getInverseViewMatrixOrthogonalX() const
      { return mInverseViewMatrixOrthogonalX; }
   const glm::mat4& getInverseViewMatrixOrthogonalY() const
      { return mInverseViewMatrixOrthogonalY; }


   glm::vec3 getEye() const            { return glm::vec3(mEye); }
   glm::vec3 getViewDirection() const  { return glm::vec3(mViewDir); }
   glm::vec3 getUpVector() const       { return glm::vec3(m_local_yAxis); }
   glm::vec3 getRightVector() const    { return glm::vec3(m_local_xAxis); }
   glm::vec3 getTarget() const;
   glm::vec3 getUserPosition() const   { return mUserPosition; }

   float getPixelSize_zNear() const { return mPixelSide_zNear; }

   void print();

private:

   void updateViewMatrix();
   void updateViewProjectionMatrix(); // projMatrix * viewMatrix
   void updateInverseViewMatrix();

   void updatePixelSize();

   // Pixel size
   float mPixelSide_zNear;

   // Matrices for this camera
   glm::mat4 mViewMatrix;
   glm::mat4 mInverseViewMatrix;
   glm::mat4 mProjectionMatrix; 
   glm::mat4 mViewProjMatrix;
   glm::mat4 mViewProjectionToUnitMatrix;
   glm::mat4 mInverseViewProjectionToUnitMatrix;

   // Matrices for orthogonal cameras
   bool mOrthogonalComputationEnabled;
   glm::mat4 mViewMatrixOrthogonalX;
   glm::mat4 mViewMatrixOrthogonalY;
   glm::mat4 mInverseViewMatrixOrthogonalX;
   glm::mat4 mInverseViewMatrixOrthogonalY;
   
   /// Frustum definition
   //@{ 
   Frustum mFrustum;
   GLfloat mAspect, mFovh;
   //@}

   GLfloat mDistanceToFrustumCenter;

   // Standard mode (looking around)
   GLfloat mAngleX; ///< Angle of rotation in degrees
   GLfloat mAngleY;   ///< Angle of rotation in degrees
   GLfloat mAngleZ;     ///< Angle of rotation in degrees

   glm::vec4 mEye;
   glm::vec4 mViewDir;
   glm::vec3 mUserPosition;

   glm::vec4 m_local_xAxis; ///< camera right vector
   glm::vec4 m_local_yAxis; ///< camera up vector
   glm::vec4 m_local_zAxis; ///< camera -viewDir vector

};

#endif
