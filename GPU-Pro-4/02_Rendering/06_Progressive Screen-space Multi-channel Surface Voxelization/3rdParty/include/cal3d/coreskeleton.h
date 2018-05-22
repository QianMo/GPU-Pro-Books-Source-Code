//****************************************************************************//
// coreskeleton.h                                                             //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_CORESKELETON_H
#define CAL_CORESKELETON_H

#include "cal3d/global.h"
#include "cal3d/refcounted.h"
#include "cal3d/refptr.h"


class CalCoreBone;
class CalCoreModel;


class CAL3D_API CalCoreSkeleton : public cal3d::RefCounted
{
protected:
  ~CalCoreSkeleton();

public:
  CalCoreSkeleton();

  int addCoreBone(CalCoreBone *pCoreBone);
  void calculateState();
  CalCoreBone* getCoreBone(int coreBoneId);
  CalCoreBone* getCoreBone(const std::string& strName);
  int getCoreBoneId(const std::string& strName);
  bool mapCoreBoneName(int coreBoneId, const std::string& strName);
  std::vector<int>& getVectorRootCoreBoneId();
  std::vector<CalCoreBone *>& getVectorCoreBone();
  void calculateBoundingBoxes(CalCoreModel * pCoreModel);
  void scale(float factor);

private:
  std::vector<CalCoreBone *> m_vectorCoreBone;
  std::map< std::string, int > m_mapCoreBoneNames;
  std::vector<int> m_vectorRootCoreBoneId;  
};
typedef cal3d::RefPtr<CalCoreSkeleton> CalCoreSkeletonPtr;

#endif

//****************************************************************************//
