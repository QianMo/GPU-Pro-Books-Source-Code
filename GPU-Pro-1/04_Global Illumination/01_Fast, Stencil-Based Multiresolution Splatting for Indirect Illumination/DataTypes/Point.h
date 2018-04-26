/******************************************************************/
/* Point.h                                                        */
/* -----------------------                                        */
/*                                                                */
/* The file defines a basic point class that implements most of   */
/*     operations you need to perform on points (multiplication,  */
/*     addition, equality testing, assignment).  NOTE:  the class */
/*     is a bit fragile when dealing with homogeneous points.  If */
/*     you only use 3D points, this will work fine for you.  If   */
/*     you use 4D points, you should read over the code and make  */
/*     sure it does what you want it to!                          */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#ifndef POINT_H
#define POINT_H 1

#include <math.h>

#include "Vector.h"
#include "Utils/TextParsing.h"

class Matrix4x4;

class Point {
  float d[4]; 
  friend class Vector;
  friend class Matrix4x4;
public:
  // Point constructors
  inline Point( const float data[3] );              // from a float array
  inline Point( float x, float y, float z );  // from individual floats
  inline Point( const Vector &v );            // copy constructor from a vector
  inline Point( const Point &v );             // copy constructor from a point
  inline Point( char *buffer );               // reads in 3 floats from the white-space delimted string
  inline Point() {}                           // default constructor

  // Mathematical operations
  inline Vector operator-(const Point& p) const;
  inline Point operator-(const Vector& v) const;
  inline Point operator+(const Vector& v) const;
  inline Point operator+(const Point&  p) const;
  inline Point& operator+=(const Vector& v);
  inline Point operator/(const float c) const;  // This should really be done with * !!
  inline Point operator*(const float c) const;
  inline Point& operator*=(const float c);
  
  // Accessor functions
  inline float X() const;
  inline float Y() const;
  inline float Z() const;
  inline float W() const                 { return 1;}
  inline float GetElement( int i ) const { return d[i]; }
  inline float *GetDataPtr()             { return d; }

  // Boolean operations
  inline bool operator != (const Point& p) const;
  inline bool operator == (const Point& p) const;

  // A 'dot product' operation involving vectors and points
  inline float Dot(const Vector &p) const;

  // Static methods.  Useful for defining commonly used points
  static Point Origin( void )  { return Point(0,0,0); }

  inline void Print( void ) const { printf( "Vec: %f %f %f\n", d[0], d[1], d[2] ); }

};


inline Point::Point( const float data[3])
{
    d[0]=data[0];
    d[1]=data[1];
    d[2]=data[2];
	d[3]=1;
}

inline Point::Point(const Vector &v) 
{
    d[0]=v.d[0]; d[1]=v.d[1]; d[2]=v.d[2]; d[3]=1;
}

inline Point::Point(const Point &v) 
{
    d[0]=v.d[0]; d[1]=v.d[1]; d[2]=v.d[2]; d[3]=1;
}

inline Point::Point(float x, float y, float z) {
    d[0]=x; d[1]=y; d[2]=z; d[3]=1;
}

inline Vector Point::operator-(const Point& p) const {
    return Vector(d[0]-p.d[0], d[1]-p.d[1], d[2]-p.d[2]);
}

inline Point Point::operator+(const Vector& v) const {
    return Point(d[0]+v.d[0], d[1]+v.d[1], d[2]+v.d[2]);
}

inline Point Point::operator+(const Point& p) const {
    return Point(d[0]+p.d[0], d[1]+p.d[1], d[2]+p.d[2]);
}

inline Point Point::operator-(const Vector& v) const {
    return Point(d[0]-v.d[0], d[1]-v.d[1], d[2]-v.d[2]);
}


inline Point Point::operator/(const float c) const {
  return Point(d[0]/c,d[1]/c,d[2]/c);
}

inline Point Point::operator*(const float c) const {
  return Point(d[0]*c,d[1]*c,d[2]*c);
}

inline Point& Point::operator*=(const float c) 
{
    d[0]=d[0]*c;
    d[1]=d[1]*c;
    d[2]=d[2]*c;
    return *this;
}


inline float Point::X() const {
    return d[0];
}

inline float Point::Y() const {
    return d[1];
}

inline float Point::Z() const {
    return d[2];
}

inline Point& Point::operator+=(const Vector& v) {
    d[0]+=v.d[0];
    d[1]+=v.d[1];
    d[2]+=v.d[2];
    return *this;
}

inline bool Point::operator != (const Point& v) const{
    return d[0] != v.d[0] || d[1] != v.d[1] || d[2] != v.d[2];
}

inline bool Point::operator == (const Point& v) const{
    return d[0] == v.d[0] && d[1] == v.d[1] && d[2] == v.d[2];
}

inline float Point::Dot(const Vector &v) const {
  return d[0]*v.d[0]+d[1]*v.d[1]+d[2]*v.d[2];
}

// Read a point from a string
inline Point::Point( char *buffer )
{
	// Default values, in case something goes wrong.
	d[0] = d[1] = d[2] = 0;  d[3]=1;

	char *ptr;
	ptr = StripLeadingNumber( buffer, &d[0] );
	ptr = StripLeadingNumber( ptr, &d[1] );
	ptr = StripLeadingNumber( ptr, &d[2] );
}

#endif

