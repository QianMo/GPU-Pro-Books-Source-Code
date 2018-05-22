//****************************************************************************//
// coordsys.h                                                                    //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_TRANSFORM_H
#define CAL_TRANSFORM_H

#include "cal3d/global.h"
#include "cal3d/vector.h"
#include "cal3d/quaternion.h"

namespace cal3d
{
  /**
   * Contains a translation and rotation that describe a coordinate
   * system or transform.
   */
  class CAL3D_API Transform
  {
  public:
    Transform() { }

    Transform(const CalVector& translation, const CalQuaternion& rotation)
    : m_translation(translation)
    , m_rotation(rotation)
    {
    }

    ~Transform() { }

    const CalVector& getTranslation() const
    {
      return m_translation;
    }

    CalVector& getTranslation()
    {
      return m_translation;
    }

    const CalQuaternion& getRotation() const
    {
      return m_rotation;
    }

    CalQuaternion& getRotation()
    {
      return m_rotation;
    }

    void setTranslation(const CalVector& translation)
    {
      m_translation = translation;
    }

    void setRotation(const CalQuaternion& rotation)
    {
      m_rotation = rotation;
    }

    /// Sets this coordinate system to the identity rotation and translation.
    void setIdentity()
    {
      m_translation.clear();
      m_rotation.clear();
    }

    void blend(float t, const Transform& end)
    {
      m_translation.blend(t, end.getTranslation());
      m_rotation.blend(t, end.getRotation());
    }

    bool operator==(const Transform& rhs) const
    {
      return m_translation == rhs.m_translation &&
             m_rotation == rhs.m_rotation;
    }

    bool operator!=(const Transform& rhs) const
    {
      return !operator==(rhs);
    }

  private:
     CalVector m_translation;
     CalQuaternion m_rotation;
  };
}

typedef cal3d::Transform CalTransform;

#endif
