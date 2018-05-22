//****************************************************************************//
// mixer.h                                                                    //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
// Copyright (C) 2004 Mekensleep <licensing@mekensleep.com>                   //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_MIXER_H
#define CAL_MIXER_H

//****************************************************************************//
// Includes                                                                   //
//****************************************************************************//

#include "cal3d/global.h"

//****************************************************************************//
// Forward declarations                                                       //
//****************************************************************************//

class CalModel;
class CalAnimation;
class CalAnimationAction;
class CalAnimationCycle;




/*****************************************************************************/
/** 
 * CalAbstractMixer defines the API that CalModel relies on for
 * blending and scheduling animations. A third party mixer must
 * implement this API in order to register itself with the
 * CalModel::setAbstractMixer method. The default mixer (CalMixer) is
 * an example of such implementation.
 *
 * cal3d expects a mixer to handle two tasks : scheduling and
 * blending. Scheduling refers to everything related to time such
 * as when an animation must run or when it must stop. Blending
 * defines how concurrent animations influence each other: for 
 * instance walking and waving.
 *
 * If CalMixer proves to be insufficient for the applications needs,
 * an alternate mixer can be implemented and used without notifying
 * cal3d in any way. It is not mandatory to subclass
 * CalAbstractMixer. However, when chosing this path, one must also
 * avoid using the CalModel::update method because it would use the
 * default mixer instantiated by the CalModel::create method with
 * undesirable side effects. In addition libraries based on cal3d
 * (think NebulaDevice or OpenSceneGraph adapters) are not aware of
 * these constraints and will keep calling the CalModel::update method of
 * CalModel regardless.
 *
 * Subclassing CalAbstractMixer when implementing an alternate mixer
 * therefore provides a better integration with cal3d and libraries
 * that rely on CalModel. However, an additional effort is required in
 * order to achieve compatibility with libraries or applications that
 * rely on the CalMixer API (i.e. that use methods such as blendCycle
 * or executeAction).  The CalMixer API is not meant to be generic and
 * there is no reason to define an abstract class that specifies
 * it. For historical reasons and because CalMixer is the default
 * mixer, some applications and libraries (think Soya or CrystalSpace)
 * depend on it. If they want to switch to a scheduler with extended
 * capabilities it might be painfull for them to learn a completely
 * different API. A scheduler with the ambition to obsolete CalMixer
 * should therefore provide an API compatible with it to ease the
 * migration process.
 *
 * Short summary, if you want to write a new mixer:
 *
 * 1) An external mixer: ignore CalAbstractMixer and implement a mixer
 * of your own. Avoid calling CalModel::update and any library or
 * application that will call it behind your back. Avoid libraries and
 * applications that rely on the default mixer CalMixer, as returned
 * by CalModel::getMixer.
 *
 * 2) A mixer registered in cal3d : subclass CalAbstractMixer,
 * register it with CalModel::setAbstractMixer.  Avoid libraries and
 * applications that rely on the default mixer CalMixer, as returned
 * by CalModel::getMixer. CalModel::getMixer will return a null
 * pointer if CalModel::setAbstractMixer was called to set
 * a mixer that is not an instance of CalMixer.
 *
 * 3) A CalMixer replacement : same as 2) and provide a subclass of
 * your own mixer that implements the CalMixer API so that existing
 * applications can switch to it by calling CalModel::getAbstractMixer
 * instead of CalModel::getMixer. The existing code using the CalMixer
 * methods will keep working and the developper will be able to 
 * switch to a new API when convenient.
 *
 *****************************************************************************/

class CAL3D_API CalAbstractMixer
{
public:
  CalAbstractMixer() {}
  virtual ~CalAbstractMixer() {}

  /*****************************************************************************/
  /**
   * Is the object an instance of the default mixer (i.e. an instance of CalMixer) ?
   *
   * @return \li \b true if an instance of CalMixer
   *         \li \b false if not an instance of CalMixer
   *
   *****************************************************************************/
  virtual bool isDefaultMixer() { return false; }

  /*****************************************************************************/
  /** 
   * Notifies the instance that updateAnimation was last called
   * deltaTime seconds ago. The internal scheduler of the instance
   * should terminate animations or update the timing information of
   * active animations accordingly. It should not blend animations
   * together or otherwise modify the CalModel associated to these
   * animations.
   *
   * The CalModel::update method will call updateSkeleton immediately
   * after updateAnimation if the instance was allocated by
   * CalModel::create (in which case it is a CalMixer instance) or if
   * the instance was set via CalModel::setAbstractMixer.
   *
   * @param deltaTime The elapsed time in seconds since the last call.
   *
   *****************************************************************************/
  virtual void updateAnimation(float deltaTime) = 0;

  /*****************************************************************************/
  /**
   * Updates the skeleton of the corresponding CalModel (as provided to
   * the create method) to match the current animation state (as
   * updated by the last call to updateAnimation).  The tracks of each
   * active animation are blended to compute the position and
   * orientation of each bone of the skeleton. The updateAnimation
   * method should be called just before calling updateSkeleton to
   * define the set of active animations.
   *
   * The CalModel::update method will call updateSkeleton immediately
   * after updateAnimation if the instance was allocated by
   * CalModel::create (in which case it is a CalMixer instance) or if
   * the instance was set via CalModel::setAbstractMixer.
   *
   *****************************************************************************/
  virtual void updateSkeleton() = 0;
};


class CAL3D_API CalMixer : public CalAbstractMixer
{
public:
  CalMixer(CalModel* pModel);
  virtual ~CalMixer();

  virtual bool isDefaultMixer() { return true; }
  bool blendCycle(int id, float weight, float delay);
  bool clearCycle(int id, float delay);
  bool executeAction(int id, float delayIn, float delayOut, float weightTarget = 1.0f, bool autoLock=false);
  bool removeAction(int id);
  virtual void updateAnimation(float deltaTime);
  virtual void updateSkeleton();
  float getAnimationTime();
  float getAnimationDuration();
  void setAnimationTime(float animationTime);
  void setTimeFactor(float timeFactor);
  float getTimeFactor();
  CalModel *getCalModel();  
  std::vector<CalAnimation *> &getAnimationVector();
  std::list<CalAnimationAction *> &getAnimationActionList();
  std::list<CalAnimationCycle *> &getAnimationCycle();  
  
protected:
  CalModel *m_pModel;
  std::vector<CalAnimation *> m_vectorAnimation;
  std::list<CalAnimationAction *> m_listAnimationAction;
  std::list<CalAnimationCycle *> m_listAnimationCycle;
  float m_animationTime;
  float m_animationDuration;
  float m_timeFactor;
};

#endif

//****************************************************************************//
