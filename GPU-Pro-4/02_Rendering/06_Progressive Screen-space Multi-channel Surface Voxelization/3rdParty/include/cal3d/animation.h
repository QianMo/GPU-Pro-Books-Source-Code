//****************************************************************************//
// animation.h                                                                //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_ANIMATION_H
#define CAL_ANIMATION_H


#include "cal3d/global.h"


class CalCoreAnimation;
class CalModel;

class CAL3D_API CalAnimation
{
public:
  enum Type
  {
    TYPE_NONE = 0,
    TYPE_CYCLE,
    TYPE_POSE,
    TYPE_ACTION
  };

  enum State
  {
    STATE_NONE = 0,
    STATE_SYNC,
    STATE_ASYNC,
    STATE_IN,
    STATE_STEADY,
    STATE_OUT,
    STATE_STOPPED
  };

protected:
  CalAnimation(CalCoreAnimation* pCoreAnimation);
public:
    virtual ~CalAnimation() {  }

  CalCoreAnimation *getCoreAnimation();
  State getState();
  float getTime();
  Type getType();
  float getWeight();
  void setTime(float time);
  void setTimeFactor(float timeFactor);
  float getTimeFactor();

  void checkCallbacks(float animationTime,CalModel *model);
  void completeCallbacks(CalModel *model);

protected:
  void setType(Type type) {
    m_type = type;
  }

  void setState(State state) {
    m_state = state;
  }

  void setWeight(float weight) {
    m_weight = weight;
  }


private:

  CalCoreAnimation *m_pCoreAnimation;
  std::vector<float> m_lastCallbackTimes;
  Type m_type;
  State m_state;
  float m_time;
  float m_timeFactor;
  float m_weight;
};

#endif

//****************************************************************************//
