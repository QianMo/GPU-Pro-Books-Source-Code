//****************************************************************************//
// skeleton.h                                                                 //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_SKELETON_H
#define CAL_SKELETON_H

#include "cal3d/global.h"

class CalCoreSkeleton;
class CalCoreModel;
class CalBone;

class CAL3D_API CalSkeleton
{
public:
  CalSkeleton(CalCoreSkeleton* pCoreSkeleton);
  ~CalSkeleton();

  void calculateState();
  void clearState();
  bool create(CalCoreSkeleton *pCoreSkeleton);
  CalBone *getBone(int boneId) const;
  CalCoreSkeleton *getCoreSkeleton() const;
  std::vector<CalBone *>& getVectorBone();
  void lockState();
  void getBoneBoundingBox(float *min, float *max);
  void calculateBoundingBoxes();

  // DEBUG-CODE
  int getBonePoints(float *pPoints);
  int getBonePointsStatic(float *pPoints);
  int getBoneLines(float *pLines);
  int getBoneLinesStatic(float *pLines);

private:
  CalCoreSkeleton *m_pCoreSkeleton;
  std::vector<CalBone *> m_vectorBone;
  bool m_isBoundingBoxesComputed;
};

#endif

//****************************************************************************//
