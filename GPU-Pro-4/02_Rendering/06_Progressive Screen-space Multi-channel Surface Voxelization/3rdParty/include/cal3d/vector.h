//****************************************************************************//
// vector.h                                                                   //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_VECTOR_H
#define CAL_VECTOR_H

//****************************************************************************//
// Includes                                                                   //
//****************************************************************************//

#include "cal3d/global.h"
#include "cal3d/matrix.h"

//****************************************************************************//
// Forward declarations                                                       //
//****************************************************************************//

class CalQuaternion;
//class CalMatrix;

//****************************************************************************//
// Class declaration                                                          //
//****************************************************************************//

 /*****************************************************************************/
/** The vector class.
  *****************************************************************************/

class CAL3D_API CalVector
{
// member variables
public:
  float x ,y ,z;

// constructors/destructor
public:
  inline CalVector(): x(0.0f), y(0.0f), z(0.0f) {};
  inline CalVector(const CalVector& v) : x(v.x), y(v.y), z(v.z) {};
  inline CalVector(float vx, float vy, float vz): x(vx), y(vy), z(vz) {};
  inline ~CalVector() {};

// member functions
public:
  inline float& operator[](unsigned int i) 
  {
	  return (&x)[i];
  }

  inline const float& operator[](unsigned int i) const
  {
	  return (&x)[i];
  }

  inline void operator=(const CalVector& v)
  {
	  x = v.x;
	  y = v.y;
	  z = v.z;
  }

  inline void operator+=(const CalVector& v)
  {
	  x += v.x;
	  y += v.y;
	  z += v.z;
  }
  
  
  inline void operator-=(const CalVector& v)
  {
	  x -= v.x;
	  y -= v.y;
	  z -= v.z;
  }

  inline void operator*=(const float d)
  {
	  x *= d;
	  y *= d;
	  z *= d;
  }

  void operator*=(const CalQuaternion& q);

  inline void operator*=(const CalMatrix &m)
  {
	  float ox = x;
	  float oy = y;
	  float oz = z;
	  x = m.dxdx*ox + m.dxdy*oy + m.dxdz*oz;
	  y = m.dydx*ox + m.dydy*oy + m.dydz*oz;
	  z = m.dzdx*ox + m.dzdy*oy + m.dzdz*oz;
  }  

  inline void operator/=(const float d)
  {
	  x /= d;
	  y /= d;
	  z /= d;
  }

  inline bool operator==(const CalVector& v) const
  {
	  return ((x == v.x) && (y == v.y) && (z == v.z));
  }

  inline bool operator!=(const CalVector& v) const
  {
    return !operator==(v);
  }

  inline void blend(float d, const CalVector& v)
  {
	  x += d * (v.x - x);
	  y += d * (v.y - y);
	  z += d * (v.z - z);
  }

  inline void clear() 
  {
	  x=0.0f;
	  y=0.0f;
	  z=0.0f;		  
  }

  inline float length() const
  {
	  return (float)sqrt(x * x + y * y + z * z);
  }
  inline float normalize()
  {
	  // calculate the length of the vector
	  float length;
	  length = (float) sqrt(x * x + y * y + z * z);
	  
	  // normalize the vector
	  x /= length;
	  y /= length;
	  z /= length;
	  
	  return length;
  }
  
  void set(float vx, float vy, float vz)
  {
	  x = vx;
	  y = vy;
	  z = vz;
  }

};

static inline CalVector operator+(const CalVector& v, const CalVector& u)
{
  return CalVector(v.x + u.x, v.y + u.y, v.z + u.z);
}

static inline CalVector operator-(const CalVector& v, const CalVector& u)
{
	return CalVector(v.x - u.x, v.y - u.y, v.z - u.z);
}

static inline CalVector operator*(const CalVector& v, const float d)
{
	return CalVector(v.x * d, v.y * d, v.z * d);
}

static inline CalVector operator*(const float d, const CalVector& v)
{
	return CalVector(v.x * d, v.y * d, v.z * d);
}

static inline CalVector operator/(const CalVector& v, const float d)
{
	return CalVector(v.x / d, v.y / d, v.z / d);
}

static inline float operator*(const CalVector& v, const CalVector& u)
{
	return v.x * u.x + v.y * u.y + v.z * u.z;
}  

static inline CalVector operator%(const CalVector& v, const CalVector& u)
{
	return CalVector(v.y * u.z - v.z * u.y, v.z * u.x - v.x * u.z, v.x * u.y - v.y * u.x);
}


 /*****************************************************************************/
/** The plane class.
  *****************************************************************************/


class CAL3D_API CalPlane
{
   public:
      float a,b,c,d;
      
      // These methods are made only to calculate the bounding boxes,
      // don't use them in you program
      
      float eval(CalVector &p);
	  float dist(CalVector &p);
      void setPosition(CalVector &p);
      void setNormal(CalVector &p);
};

 /*****************************************************************************/
/** The bounding box class.
  *****************************************************************************/


class CAL3D_API CalBoundingBox
{
   public:
     CalPlane plane[6];
     
     void computePoints(CalVector *p);
   
};



#endif

//****************************************************************************//
