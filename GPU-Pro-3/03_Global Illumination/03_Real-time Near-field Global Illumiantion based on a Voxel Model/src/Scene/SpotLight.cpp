#include "SpotLight.h"

#include <iostream>

const glm::mat4 SpotLight::mScaleBiasMatrix = glm::translate(0.5f, 0.5f, 0.5f) * glm::scale(0.5f, 0.5f, 0.5f);
int SpotLight::mSpotMapResolution = 1024;

SpotLight::SpotLight(glm::vec3 worldPosition,
                     glm::vec3 I,
                     float cutoffAngle,
                     float spotExponent, 
                     float constantAttenuation,
                     float quadraticAttenuation,
                     float angleX, float angleY, float angleZ)
                     :mWorldPosition(worldPosition),
                       mI(I),
                       mConstantAttenuation(constantAttenuation),
                       mQuadraticAttenuation(quadraticAttenuation),
                     mAngleX(angleX), mAngleY(angleY), mAngleZ(angleZ)
{
   zNear = 0.01f;

   setCutoffAngle(cutoffAngle);
   setExponent(spotExponent);
   
   mChanged = true;

   // init local light coordinate system 
   m_local_xAxis = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
   m_local_yAxis = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
   m_local_zAxis = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);

   // standard spot light's cone points along negative z-axis
   mSpotDirection = glm::vec3(-m_local_zAxis);

   updateViewMatrix();
   updateProjMatrix();
}

SpotLight::SpotLight(const SpotLight* const copySpot)
: mWorldPosition(copySpot->mWorldPosition),
                       mI(copySpot->mI),
                       mConstantAttenuation(copySpot->mConstantAttenuation),
                       mQuadraticAttenuation(copySpot->mQuadraticAttenuation),
  mAngleX(copySpot->mAngleX), mAngleY(copySpot->mAngleY), mAngleZ(copySpot->mAngleZ)
{
   setCutoffAngle(copySpot->mCutoffAngle);
   setExponent(copySpot->mExponent);
   zNear = copySpot->zNear;
   
   mChanged = true;

   // init local light coordinate system 
   m_local_xAxis = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
   m_local_yAxis = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
   m_local_zAxis = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);

   // standard spot light's cone points along negative z-axis
   mSpotDirection = glm::vec3(-m_local_zAxis);

   void updateViewMatrix();
   void updateMapLookupMatrix();
   void updateProjMatrix();

   modifyCutoffAngle(static_cast<float>(rand() % 10));
   updatePixelSide();
}

void SpotLight::print()
{
   std::cout << "Position x: " << mWorldPosition.x << std::endl;
   std::cout << "         y: " << mWorldPosition.y << std::endl;
   std::cout << "         z: " << mWorldPosition.z << std::endl;

   std::cout << "Direction x: " << mSpotDirection.x << std::endl;
   std::cout << "          y: " << mSpotDirection.y << std::endl;
   std::cout << "          z: " << mSpotDirection.z << std::endl;

   std::cout << "Angle x: " << mAngleX << std::endl;
   std::cout << "Angle y: " << mAngleY << std::endl;
   std::cout << "Angle z: " << mAngleZ << std::endl;

}

void SpotLight::updatePixelSide()
{
   float right = getFrustumRight();

   mPixelSide = 2 * right / mSpotMapResolution;
      
   // mPixelSide / zNear =
   mPixelSide_zNear = mPixelSide / zNear;

   mPixelDiag = sqrt(2.0f) * mPixelSide;
   mPixelDiag_zNear = sqrt(2.0f) * mPixelSide_zNear;

   //std::cout << "PixelSide (comp 1) = " << pixelSide_zNear << std::endl;
   //std::cout << "PixelSide (comp 2) = " << (2.0f * tan(cutoffAngle * PI / 360.0f) / mSpotMapResolution) << std::endl;

   // identical results :)
   // PixelSide (comp 1) = 0.0021666
   // PixelSide (comp 2) = 0.0021666
   
   // or:
   // pixelSide_zNear = 2.0f * tan(cutoffAngle * PI / 360.0f) / mSpotMapResolution;
   // because of:
   //     frustum.right         = tan(fovh * PI / 360.0f) * zNear;
   // <=> frustum.right / zNear = tan(fovh * PI / 360.0f)


}

void SpotLight::updateProjMatrix()
{
// assume aspect = 1.0, i.e. a square texture shadow map
   mProjMatrix = glm::perspective(mCutoffAngle * 2.0f, 1.0f, zNear, 20.0f);

   updateMapLookupMatrix();
   updatePixelSide();
}

void SpotLight::updateViewMatrix()
{
   glm::mat4 rotationMatrix;
   rotationMatrix = glm::rotate(glm::mat4(1.0), -mAngleZ, WORLD_ZAXIS);
   rotationMatrix = glm::rotate(rotationMatrix, -mAngleY, WORLD_YAXIS);
   rotationMatrix = glm::rotate(rotationMatrix, -mAngleX, WORLD_XAXIS);

   m_local_xAxis = rotationMatrix * glm::vec4(WORLD_XAXIS, 0.0f);
   m_local_yAxis = rotationMatrix * glm::vec4(WORLD_YAXIS, 0.0f);
   m_local_zAxis = rotationMatrix * glm::vec4(WORLD_ZAXIS, 0.0f);

   mSpotDirection = glm::normalize(glm::vec3(-m_local_zAxis));

   mLightViewMatrix = glm::lookAt(mWorldPosition, mWorldPosition + mSpotDirection, glm::vec3(m_local_yAxis));

   mInverseLightViewMatrix = glm::inverse(mLightViewMatrix);

   updateMapLookupMatrix();

}


void SpotLight::updateMapLookupMatrix()
{
   // compose matrix from scale-bias + perspective + view
   mMapLookupMatrix = mScaleBiasMatrix * mProjMatrix * mLightViewMatrix;
   mChanged = true;

}

void SpotLight::setI(float r, float g, float b)
{
   mI.x = r;
   mI.y = g;
   mI.z = b;
}

void SpotLight::scaleI(float factor)
{
   mI.x *= factor;
   mI.y *= factor;
   mI.z *= factor;
}


void SpotLight::move(float dx, float dy, float dz)
{
   mWorldPosition += dz * mSpotDirection;
   mWorldPosition += dy * glm::vec3(m_local_yAxis);
   mWorldPosition += dx * glm::vec3(m_local_xAxis);

   updateViewMatrix();
}

void SpotLight::setPosition(float x, float y, float z)
{
   mWorldPosition.x = x;
   mWorldPosition.y = y;
   mWorldPosition.z = z;

   updateViewMatrix();
}

void SpotLight::setPosition(glm::vec3 newPosition)
{
   mWorldPosition = newPosition;
   updateViewMatrix();
}


void SpotLight::setCutoffAngle(float angle)
{
   mCutoffAngle = angle;
   mCosCutoff = cos(angle / 180.0f * F_PI);
   mInnerCosCutoff = cos(angle * 0.95f / 180.0f * F_PI);
   mSolidAngle = 2.0f * F_PI * (1.0f - mCosCutoff);
   updateProjMatrix(); 
}

void SpotLight::modifyCutoffAngle(float deltaAngle)
{
   if(mCutoffAngle + deltaAngle >= 2 && mCutoffAngle + deltaAngle < 80)
   {
      setCutoffAngle(mCutoffAngle + deltaAngle);
   }
}


void SpotLight::setAngleX(GLfloat angle)
{
   mAngleX = angle;
   updateViewMatrix();
}

void SpotLight::setAngleY(GLfloat angle)
{
   mAngleY = angle;
   updateViewMatrix();
}

void SpotLight::setAngleZ(GLfloat angle)
{
   mAngleZ = angle;
   updateViewMatrix();
}

void SpotLight::modifyAngles(float angleXDelta, float angleYDelta, float angleZDelta)
{
   mAngleX += angleXDelta;
   mAngleY += angleYDelta;
   mAngleZ += angleZDelta;
   updateViewMatrix();
}

void SpotLight::modifyAngleX(GLfloat delta)
{
   mAngleX += delta;
   updateViewMatrix();

}
void SpotLight::modifyAngleY(GLfloat delta)
{
   mAngleY += delta;
   updateViewMatrix();
}
void SpotLight::modifyAngleZ(GLfloat delta)
{
   mAngleZ += delta;
   updateViewMatrix();
}