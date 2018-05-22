//****************************************************************************//
// bone.h                                                                     //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_BONE_H
#define CAL_BONE_H


#include "cal3d/global.h"
#include "cal3d/vector.h"
#include "cal3d/quaternion.h"
#include "cal3d/matrix.h"
#include "cal3d/corebone.h"


//class CalCoreBone;
class CalSkeleton;
class CalModel;
class CalCoreModel;


class CAL3D_API CalBone
{
public:
  CalBone(CalCoreBone* coreBone);
  ~CalBone() { }

  void blendState(float weight, const CalVector& translation, const CalQuaternion& rotation);
  void calculateState();
  void clearState();
  CalCoreBone *getCoreBone();
  void setCoreState();
  void setCoreStateRecursive();
  void setRotation(const CalQuaternion& rotation);
  const CalQuaternion& getRotation();
  const CalQuaternion& getRotationAbsolute();
  const CalQuaternion& getRotationBoneSpace();
  void setTranslation(const CalVector& translation);
  const CalVector& getTranslation();
  const CalVector& getTranslationAbsolute();
  const CalVector& getTranslationBoneSpace();
  const CalMatrix& getTransformMatrix();
  void lockState();
  void setSkeleton(CalSkeleton *pSkeleton);
  void calculateBoundingBox();
  CalBoundingBox & getBoundingBox();

private:
  CalCoreBone *m_pCoreBone;
  CalSkeleton *m_pSkeleton;
  float m_accumulatedWeight;
  float m_accumulatedWeightAbsolute;
  CalVector m_translation;
  CalQuaternion m_rotation;
  CalVector m_translationAbsolute;
  CalQuaternion m_rotationAbsolute;
  CalVector m_translationBoneSpace;
  CalQuaternion m_rotationBoneSpace;
  CalMatrix m_transformMatrix;  
  CalBoundingBox m_boundingBox;
};

#endif

//****************************************************************************//
