//****************************************************************************//
// animation_action.h                                                         //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_ANIMATION_ACTION_H
#define CAL_ANIMATION_ACTION_H


#include "cal3d/global.h"
#include "cal3d/animation.h"


class CalCoreAnimation;


class CAL3D_API CalAnimationAction : public CalAnimation
{
public:
  CalAnimationAction(CalCoreAnimation* pCoreAnimation);
  virtual ~CalAnimationAction() { }

  bool execute(float delayIn, float delayOut, float weightTarget = 1.0f,bool autoLock=false);
  bool update(float deltaTime);

private:
  float m_delayIn;
  float m_delayOut;
  float m_delayTarget;
  float m_weightTarget;
  bool  m_autoLock; 
};

#endif
