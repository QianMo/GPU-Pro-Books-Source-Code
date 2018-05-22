//****************************************************************************//
// animcallback.h                                                             //
// Copyright (C) 2004  Keith Fulton <keith@paqrat.com>                        //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_ANIMCALLBACK_H
#define CAL_ANIMCALLBACK_H


#include "cal3d/global.h"


class CalAnimation;

struct CalAnimationCallback
{
    virtual ~CalAnimationCallback() {}
    virtual void AnimationUpdate(float anim_time,CalModel *model, void * userData) = 0;
    virtual void AnimationComplete(CalModel *model, void * userData) = 0;
};


#endif

//****************************************************************************//
