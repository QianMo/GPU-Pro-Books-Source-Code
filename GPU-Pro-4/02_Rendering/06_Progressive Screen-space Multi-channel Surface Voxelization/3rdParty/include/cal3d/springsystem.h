//****************************************************************************//
// springsystem.h                                                             //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_SPRINGSYSTEM_H
#define CAL_SPRINGSYSTEM_H

//****************************************************************************//
// Includes                                                                   //
//****************************************************************************//

#include "cal3d/global.h"
#include "cal3d/vector.h"

//****************************************************************************//
// Forward declarations                                                       //
//****************************************************************************//

class CalModel;
class CalSubmesh;

//****************************************************************************//
// Class declaration                                                          //
//****************************************************************************//

 /*****************************************************************************/
/** The spring system class.
  *****************************************************************************/

class CAL3D_API CalSpringSystem
{
public:
  CalSpringSystem(CalModel* pModel);
  ~CalSpringSystem() { }

// member functions	
public:
  void calculateForces(CalSubmesh *pSubmesh, float deltaTime);
  void calculateVertices(CalSubmesh *pSubmesh, float deltaTime);
  void update(float deltaTime);
  
  CalVector & getGravityVector();
  void setGravityVector(const CalVector & vGravity);
  CalVector & getForceVector();
  void setForceVector(const CalVector & vForce);
  void setCollisionDetection(bool collision);


  /* DEBUG CODE ********************
  struct
  {
    float x, y, z, radius;
  } Sphere;
  void setSphere(float x, float y, float z, float radius) { Sphere.x = x; Sphere.y = y; Sphere.z = z; Sphere.radius = radius; };
  *********************************/

private:
  CalModel *m_pModel;
  CalVector m_vGravity;  
  CalVector m_vForce;  
  bool m_collision;
};

#endif

//****************************************************************************//
