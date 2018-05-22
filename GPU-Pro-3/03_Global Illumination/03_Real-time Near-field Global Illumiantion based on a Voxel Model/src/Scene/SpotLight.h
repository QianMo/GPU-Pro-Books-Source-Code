#ifndef SPOTLIGHT_H
#define SPOTLIGHT_H

#include "OpenGL.h"

#include "glm/gtc/matrix_transform.hpp" // for glm::translate, glm::rotate, glm::scale
#include "glm/gtc/matrix_projection.hpp" // for perspective projection
#include "glm/gtx/transform2.hpp" // for glm::lookAt 

class SpotLight
{
public:
   SpotLight(glm::vec3 worldPosition = glm::vec3(0, 1, 0),
      glm::vec3 I = glm::vec3(1, 1, 1),
      float cutoffAngle = 45.0f,
      float spotExponent = 1.0f,
      float constantAttenuation = 1.0f,
      float quadraticAttenuation = 0.0f,
      float angleX = 0.0f, float angleY = 0.0f, float angleZ = 0.0f
      );

   SpotLight(const SpotLight* const copySpot);

   // overwrite point light method
   void print();
   void setPosition(glm::vec3 newPosition);
   void setPosition(float x, float y, float z);

   void setCutoffAngle(float angle);
   void modifyCutoffAngle(float deltaAngle);
   void setExponent(float e) { mExponent = e; }
   static void changeResolution(int res) { mSpotMapResolution = res; }

   const glm::vec3& getSpotDirection() const { return mSpotDirection; }
   glm::vec3 getSavedSpotDirection() const   { return -glm::vec3(m_local_zAxis); };

   glm::vec3 getPhi() const { return mI * mSolidAngle; } 
   float getConstantAttenuation() const { return mConstantAttenuation; }
   float getQuadraticAttenuation() const { return mQuadraticAttenuation; }
   /// Returns luminosity [german Lichtstaerke] in [candela]
   const glm::vec3& getI() const { return mI; };
   /// Returns light position in world space
   const glm::vec3& getPosition() const { return mWorldPosition; }

   /// Setter
   void setI(glm::vec3 newI)               { mI = newI; }
   void setI(float r, float g, float b);
   void scaleI(float factor);
   void setConstantAttenuation(float att)  { mConstantAttenuation = att; }
   void setQuadraticAttenuation(float att) { mQuadraticAttenuation = att; }

   glm::vec3 getUpVector() const        { return glm::vec3(m_local_yAxis); }
   float getCosCutoffAngle() const      { return mCosCutoff; } // outer
   float getInnerCosCutoffAngle() const { return mInnerCosCutoff; }
   float getCutoffAngle() const         { return mCutoffAngle; }
   float getSolidAngle() const          { return mSolidAngle; }
   float getExponent() const            { return mExponent; }
   static int spotMapResolution()       { return mSpotMapResolution; }

   /// Pixel sizes of spot map in world coordinates
   float getPixelSide() const { return mPixelSide; }
   float getPixelDiag() const { return mPixelDiag; }
   float getPixelSide_zNear() const { return mPixelSide_zNear; }
   float getPixelDiag_zNear() const { return mPixelDiag_zNear; }
   float getZNear() const { return zNear; }

   bool lookupMatrixChanged() const { return mChanged; }
   void setChanged(bool c)          { mChanged = c; }

   const glm::mat4& getMapLookupMatrix() const { return mMapLookupMatrix; }
   const glm::mat4& getLightViewMatrix() const { return mLightViewMatrix; }
   const glm::mat4& getInverseLightViewMatrix() const { return mInverseLightViewMatrix; }
   const glm::mat4& getProjectionMatrix() const { return mProjMatrix; }

   /// Set angle for rotating around x-axis in degrees.
   void setAngleX(float angle);
   /// Set angle for rotating around y-axis in degrees. 
   void setAngleY(float angle);
   /// Set angle for rotating around z-axis in degrees.
   void setAngleZ(float angle);

   float getAngleX() const { return mAngleX; }
   float getAngleY() const { return mAngleY; }
   float getAngleZ() const { return mAngleZ; }

   /// Modifies value by adding delta.
   //@{
   void modifyAngles(float angleXDelta, float angleYDelta, float angleZDelta);
   void modifyAngleX(float delta);
   void modifyAngleY(float delta);
   void modifyAngleZ(float delta);
   //@}

   void move(float dx, float dy, float dz);

   void updatePixelSide();

   static const glm::mat4& getScaleBiasMatrix() { return mScaleBiasMatrix; }

private:
   float getFrustumRight() const { return tan(mCutoffAngle * F_PI / 180.0f) * zNear; }

   void updateViewMatrix();
   void updateMapLookupMatrix();
   void updateProjMatrix();

   glm::vec3 mWorldPosition;
   glm::vec3 mI; // [candela] rgb

   // Scale factors for attenuation
   // att = 1.0 / ( constScale + quadrScale * d * d)
   float mConstantAttenuation;  
   float mQuadraticAttenuation; 

   float zNear;
   float mPixelSide; ///< depends on frustum (cone angle and zNear) and SpotMap resolution
   float mPixelSide_zNear; ///< = pixelSide / zNear
   float mPixelDiag; ///< the diagonal of a pixel with sides pixelSide
   float mPixelDiag_zNear; ///< = pixelDiag / zNear

   glm::mat4 mMapLookupMatrix; ///< For shadow mapping (shader)
   glm::mat4 mLightViewMatrix; ///< For shadow mapping (rendering of shadow map)
   glm::mat4 mInverseLightViewMatrix;
   glm::mat4 mProjMatrix;
   static const glm::mat4 mScaleBiasMatrix;

   // Spot map (RSM) resolution
   static int mSpotMapResolution;

   bool mChanged;

   glm::vec4 m_local_xAxis; ///< spot camera right vector
   glm::vec4 m_local_yAxis; ///< spot camera up vector
   glm::vec4 m_local_zAxis; ///< spot camera -viewDir vector

   float mAngleX, mAngleY, mAngleZ; ///< defines spot direction
   glm::vec3 mSpotDirection; ///< Axis of the cone of light (normalized)

   float mCutoffAngle;   ///< In degrees. Cone's apex angle is 2*cutoffAngle.
   float mCosCutoff; ///< cosinus of (outer) cutoff angle
   float mInnerCosCutoff; ///< cosinus of (inner) cutoff angle
   float mExponent; ///< controls how concentrated the light is

   float mSolidAngle; ///< solid angle of spot light cone, computed from cutoffAngle

   // spotExponent Explanation (red book):
   // The light's intensity is highest in the center of the cone.
   // It is attenuated toward the edges of the cone by the cosinus
   // of the angle between the spot direction and the direction from the light
   // to the vertex being lit, raised to the power of the spot exponent.

};

#endif
