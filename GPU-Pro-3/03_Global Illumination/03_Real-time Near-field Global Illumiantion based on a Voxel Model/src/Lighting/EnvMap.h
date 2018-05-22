#ifndef ENVMAP_H
#define ENVMAP_H

#include "OpenGL.h"

#include <string>
#include <iostream>

using namespace std;

class EnvMap
{
public:
   EnvMap(string fileName);

   void rotate(float dx);

   GLuint getTexture() const { return mEnvMapTextureId; }
   int getTextureWidth() const { return mEnvMapWidth; }
   int getTextureHeight() const { return mEnvMapHeight; }
   float getRotationAngle() const { return mRotationAngle; }
   void setRotationAngle(float angle) { mRotationAngle = angle; }
   void setRotationAngle(int angleInDegrees) { mRotationAngle = angleInDegrees / 180.0f * 3.14159f; }

   bool loadPFM(string fileName);

private:
   EnvMap();

   void createTextureFromLoadedPFM();

   GLuint mEnvMapTextureId;

   int mEnvMapWidth, mEnvMapHeight; // width and height of env map
   float* mEnvMapPixels;

   float mInvGamma; // inverse gamma
   float mLMax; // maximum luminance
   float mRotationAngle;     // envmap rotation around y axis            

};

#endif
