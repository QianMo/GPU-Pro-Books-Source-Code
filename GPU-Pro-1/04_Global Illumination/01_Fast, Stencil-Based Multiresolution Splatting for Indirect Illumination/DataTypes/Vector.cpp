/******************************************************************/
/* Vector.cpp                                                     */
/* -----------------------                                        */
/*                                                                */
/* The file defines a basic vector class that implments most of   */
/*     operations you need to perform on vectors (multiplication, */
/*     addition, equality testing, assignment).  NOTE:  the class */
/*     is a bit fragile when dealing with homogeneous coords.  If */
/*     you only use 3D vectors, this will work fine for you.  If  */
/*     you use 4D vectors, you should read over the code and make */
/*     sure it does what you want it to!                          */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#include "Vector.h"
#include "Point.h"

Vector::Vector(const Point& v) 
{
    d[0]=v.d[0];
    d[1]=v.d[1];
    d[2]=v.d[2];
}

Vector& Vector::operator=(const Point& p) 
{
  d[0] = p.d[0];
  d[1] = p.d[1];
  d[2] = p.d[2];
  return *this;
}

float Vector::Dot(const Point& p) const 
{
    return d[0]*p.d[0]+d[1]*p.d[1]+d[2]*p.d[2];
}

Vector Vector::operator+(const Point& p) const 
{
    return Vector(d[0]+p.d[0], d[1]+p.d[1], d[2]+p.d[2]);
}

Vector Vector::operator-(const Point& p) const 
{
    return Vector(d[0]-p.d[0], d[1]-p.d[1], d[2]-p.d[2]);
}

