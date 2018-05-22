//****************************************************************************//
// corekeyframe.h                                                             //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_COREKEYFRAME_H
#define CAL_COREKEYFRAME_H

//****************************************************************************//
// Includes                                                                   //
//****************************************************************************//

#include "cal3d/global.h"
#include "cal3d/matrix.h"
#include "cal3d/vector.h"
#include "cal3d/quaternion.h"

//****************************************************************************//
// Class declaration                                                          //
//****************************************************************************//

 /*****************************************************************************/
/** The core keyframe class.
  *****************************************************************************/

class CAL3D_API CalCoreKeyframe
{
// member variables
protected:
  float m_time;
  CalVector m_translation;
  CalQuaternion m_rotation;

public:
// constructors/destructor
  CalCoreKeyframe();
  virtual ~CalCoreKeyframe();

// member functions
public:
  bool create();
  void destroy();
  const CalQuaternion& getRotation();

  /*****************************************************************************/
  /** Returns the time.
  *
  * This function returns the time of the core keyframe instance.
  *
  * @return The time in seconds.
  *****************************************************************************/
  inline float getTime() const
  {
	  return m_time;
  }

  const CalVector& getTranslation();
  void setRotation(const CalQuaternion& rotation);
  void setTime(float time);
  void setTranslation(const CalVector& translation);
};

#endif

//****************************************************************************//
