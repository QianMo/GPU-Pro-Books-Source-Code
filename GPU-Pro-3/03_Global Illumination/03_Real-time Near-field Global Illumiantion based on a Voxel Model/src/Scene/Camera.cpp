#include "Camera.h"

#include "Utils/CoutMethods.h" // for coutMatrix
#include "Scene/Scene.h"

#include "glm/gtc/matrix_transform.hpp" // for glm::translate, glm::rotate, glm::scale
#include "glm/gtc/matrix_projection.hpp" // for projection matrix
#include "glm/gtx/transform2.hpp" // for glm::lookAt

Camera::Camera()
{ 
   mDistanceToFrustumCenter = 2.0; // some default value for perspective projection

   mOrthogonalComputationEnabled = false;

   // init local camera coordinate system 
   m_local_xAxis = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
   m_local_yAxis = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
   m_local_zAxis = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);

   mViewMatrix = glm::mat4(1.0); // identity

   // standard cam sits in origin and looks along negative z-axis
   mEye = glm::vec4(0, 0, 0, 1);
   mUserPosition = glm::vec3(0);
   mViewDir = -m_local_zAxis;

   setAngleX(0.0f);
   setAngleY(0.0f);
   setAngleZ(0.0f);
}


Camera::~Camera()
{
}

void Camera::setFovH(float newFovH)
{
   setPerspectiveFrustum(newFovH, mAspect, mFrustum.zNear, mFrustum.zFar);
}

void Camera::setZNear(float newZNear)
{
   setPerspectiveFrustum(mFovh, mAspect, newZNear, mFrustum.zFar);
}

void Camera::setPerspectiveFrustum(GLfloat fovh, GLfloat aspect, GLfloat zNear, GLfloat zFar)
{
   //mFrustum.top    = tan(fovy * PI / 360.0f) * zNear;
   //mFrustum.bottom = -mFrustum.top;
   //mFrustum.left   = mAspect * mFrustum.bottom;
   //mFrustum.right  = mAspect * mFrustum.top;
   
   // Update frustum data
   this->mAspect = aspect;
   this->mFovh   = fovh;

   mFrustum.right    = tan(fovh * F_PI / 360.0f) * zNear;
   mFrustum.left     = -mFrustum.right;
   mFrustum.top      = mFrustum.right / aspect;
   mFrustum.bottom   = -mFrustum.top;

   mFrustum.zNear  = zNear;
   mFrustum.zFar   = zFar;
   mFrustum.width  = mFrustum.right - mFrustum.left;
   mFrustum.height = mFrustum.top   - mFrustum.bottom;
   mFrustum.zRange = mFrustum.zFar  - mFrustum.zNear;

   mProjectionMatrix = glm::frustum(mFrustum.left, mFrustum.right,
                                   mFrustum.bottom, mFrustum.top,
                                   mFrustum.zNear, mFrustum.zFar);


   updateViewProjectionMatrix();

}

void Camera::setOrthographicFrustum(GLfloat left, GLfloat right,
                                          GLfloat bottom, GLfloat top,
                                          GLfloat zNear, GLfloat zFar)
{
   mFrustum.left   = left;
   mFrustum.right  = right;
   mFrustum.bottom = bottom;
   mFrustum.top    = top;
   mFrustum.zNear  = zNear;
   mFrustum.zFar   = zFar;
   mFrustum.width  = right - left;
   mFrustum.height = top   - bottom;
   mFrustum.zRange = zFar  - zNear;


   mProjectionMatrix = glm::ortho(mFrustum.left, mFrustum.right,
                                 mFrustum.bottom, mFrustum.top,
                                 mFrustum.zNear, mFrustum.zFar);


   mDistanceToFrustumCenter = mFrustum.zNear + mFrustum.zRange / 2.0f;

   updateViewProjectionMatrix();
}

void Camera::updateViewProjectionMatrix()
{
   mViewProjMatrix = mProjectionMatrix * mViewMatrix;
   mViewProjectionToUnitMatrix = glm::translate(0.5f, 0.5f, 0.5f)  * glm::scale(0.5f, 0.5f, 0.5f) *  mProjectionMatrix * mViewMatrix;
   mInverseViewProjectionToUnitMatrix = glm::inverse(mViewProjectionToUnitMatrix);

   updatePixelSize();
}

void Camera::updatePixelSize()
{
   // camera and pixel sizes
   float windowPixelSize  = mFrustum.width / Scene::Instance()->getWindowWidth();

   //cout << endl << endl << "WINDOW PIXEL SIZE INFO " << endl << endl ;
   //cout << "windowPixelSize " << windowPixelSize << endl;
      
   // pixelSide / zNear =
   mPixelSide_zNear = windowPixelSize / mFrustum.zNear;


}


void Camera::updateViewMatrix()
{

   // update eye
   mEye = glm::vec4(mUserPosition, 0);
               
   // update viewing direction and view matrix
   glm::mat4 rotationMatrix; 
   rotationMatrix = glm::rotate(glm::mat4(1.0), -mAngleZ, WORLD_ZAXIS);
   rotationMatrix = glm::rotate(rotationMatrix, -mAngleY, WORLD_YAXIS);
   rotationMatrix = glm::rotate(rotationMatrix, -mAngleX, WORLD_XAXIS);

   m_local_xAxis = rotationMatrix * glm::vec4(WORLD_XAXIS, 0.0f);
   m_local_yAxis = rotationMatrix * glm::vec4(WORLD_YAXIS, 0.0f);
   m_local_zAxis = rotationMatrix * glm::vec4(WORLD_ZAXIS, 0.0f);

   mViewDir = - m_local_zAxis;

   mViewMatrix = glm::lookAt(glm::vec3(mEye),
      glm::vec3(mEye + mViewDir),
      glm::vec3(m_local_yAxis));

   updateInverseViewMatrix();

   if(mOrthogonalComputationEnabled)
   {
      glm::vec4 t = mEye + mDistanceToFrustumCenter * mViewDir;

      mViewMatrixOrthogonalX = mViewMatrix
         * glm::translate(glm::vec3(t))
         * glm::rotate(glm::mat4(1.0), -90.0f, glm::vec3(m_local_yAxis))
         * glm::translate(glm::vec3(-t));

      mViewMatrixOrthogonalY = mViewMatrix
         * glm::translate(glm::vec3(t))
         * glm::rotate(glm::mat4(1.0), 90.0f, glm::vec3(m_local_xAxis))
         * glm::translate(glm::vec3(-t));


      mInverseViewMatrixOrthogonalX = glm::inverse(mViewMatrixOrthogonalX);
      mInverseViewMatrixOrthogonalY = glm::inverse(mViewMatrixOrthogonalY);

   }

   updateViewProjectionMatrix();

}


void Camera::updateInverseViewMatrix()
{
   mInverseViewMatrix = glm::inverse(mViewMatrix);
}

void Camera::enableOrthogonalViews()
{
   if(!mOrthogonalComputationEnabled)
   {
      mOrthogonalComputationEnabled = true;
      mViewMatrixOrthogonalX = glm::mat4(1.0);
      mViewMatrixOrthogonalY = glm::mat4(1.0);
      updateViewMatrix();
   }
}


void Camera::setUserPosition(float x, float y, float z)
{
   mUserPosition.x = x;
   mUserPosition.y = y;
   mUserPosition.z = z;
   updateViewMatrix();
}

void Camera::setUserPosition(glm::vec3 newP)
{
   setUserPosition(newP.x, newP.y, newP.z);
}

void Camera::setAngleX(GLfloat angle)
{
   mAngleX = angle;
   updateViewMatrix();
}

void Camera::setAngleY(GLfloat angle)
{
   mAngleY = angle;
   updateViewMatrix();
}

void Camera::setAngleZ(GLfloat angle)
{
   mAngleZ = angle;
   updateViewMatrix();
}



void Camera::move(GLfloat deltaX, GLfloat deltaY, GLfloat deltaZ)
{
   mUserPosition += deltaZ * glm::vec3(mViewDir);
   mUserPosition += deltaY * glm::vec3(m_local_yAxis);
   mUserPosition += deltaX * glm::vec3(m_local_xAxis);

   updateViewMatrix();
}

void Camera::modifyAngleX(GLfloat delta)
{
   mAngleX += delta;
   updateViewMatrix();

}
void Camera::modifyAngleY(GLfloat delta)
{
   mAngleY += delta;
   updateViewMatrix();
}
void Camera::modifyAngleZ(GLfloat delta)
{
   mAngleZ += delta;
   updateViewMatrix();
}



glm::vec3 Camera::getTarget() const
{
   return glm::vec3(mEye+mViewDir);
}

void Camera::print()
{
   cout << "Camera Eye:       " << mEye.x << " " << mEye.y << " " << mEye.z << endl;
   cout << "Camera View Dir:  " << mViewDir.x << " " << mViewDir.y << " " << mViewDir.z << endl;

   cout << "Camera angle X:   " << mAngleX << endl;
   cout << "Camera angle Y:   " << mAngleY << endl;
   cout << "Camera angle Z:   " << mAngleZ << endl;

   cout << "View Matrix: " << endl;
   coutMatrix(mViewMatrix);
}
