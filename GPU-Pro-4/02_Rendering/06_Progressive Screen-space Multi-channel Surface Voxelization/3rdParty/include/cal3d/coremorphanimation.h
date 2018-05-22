//****************************************************************************//
// coremorphanimation.h                                                       //
// Copyright (C) 2003 Steven Geens                                            //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_COREMORPHANIMATION_H
#define CAL_COREMOPRHANIMATION_H

#include "cal3d/global.h"

class CAL3D_API CalCoreMorphAnimation
{
public:
  CalCoreMorphAnimation()  { }
  ~CalCoreMorphAnimation() { }

  bool addMorphTarget(int coreMeshID,int morphTargetID);
  std::vector<int>& getVectorCoreMeshID();
  std::vector<int>& getVectorMorphTargetID();

private:
  std::vector<int> m_vectorCoreMeshID;
  std::vector<int> m_vectorMorphTargetID;
};

#endif

//****************************************************************************//
