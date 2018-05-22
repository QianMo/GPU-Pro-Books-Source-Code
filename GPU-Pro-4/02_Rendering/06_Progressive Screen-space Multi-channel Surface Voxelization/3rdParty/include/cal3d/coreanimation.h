//****************************************************************************//
// coreanimation.h                                                            //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_COREANIMATION_H
#define CAL_COREANIMATION_H

#include "cal3d/global.h"
#include "cal3d/quaternion.h"
#include "cal3d/refcounted.h"
#include "cal3d/refptr.h"

struct CalAnimationCallback;
class CalCoreTrack;

class CAL3D_API CalCoreAnimation : public cal3d::RefCounted
{
protected:
  ~CalCoreAnimation();

public:
  CalCoreAnimation();

  bool addCoreTrack(CalCoreTrack *pCoreTrack);
  CalCoreTrack *getCoreTrack(int coreBoneId);

	float getDuration() const;
  void setDuration(float duration);
  void scale(float factor);
  void setFilename(const std::string& filename);
  const std::string& getFilename(void) const;
  void setName(const std::string& name);
  const std::string& getName(void) const;

  void registerCallback(CalAnimationCallback *callback,float min_interval);
  void removeCallback(CalAnimationCallback *callback);

  unsigned int getTrackCount() const;
  std::list<CalCoreTrack *>& getListCoreTrack();
	unsigned int getTotalNumberOfKeyframes() const;

  struct CallbackRecord
  {
    CalAnimationCallback *callback;
    float                 min_interval;
  };

  std::vector<CallbackRecord>& getCallbackList() { return m_listCallbacks; }

private:

  std::vector<CallbackRecord> m_listCallbacks;

  float m_duration;
  std::list<CalCoreTrack *> m_listCoreTrack;
  std::string m_name;
  std::string m_filename;
};

typedef cal3d::RefPtr<CalCoreAnimation> CalCoreAnimationPtr;

#endif

//****************************************************************************//
