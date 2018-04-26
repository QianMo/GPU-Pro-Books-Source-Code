/******************************************************************/
/* Vector.h                                                       */
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

#ifndef VECTOR_H
#define VECTOR_H 1

#include <math.h>
#include <iostream>
#include <stdio.h>

#include "MathDefs.h"
#include "Utils/TextParsing.h"

class Point;
class Matrix4x4;

class Vector {
    float d[4]; 
    friend class Point; 
	friend class Matrix4x4;
public:
	// Vector constructors
	inline Vector(float data[3]);             // from a float array
    inline Vector(float x, float y, float z); // from individual floats
    inline Vector(const Vector& v);           // copy constructor from another vector
	inline Vector() {}                        // default constructor
	inline Vector( char *buffer );            // reads in 3 floats from the white-space delimted string
	Vector(const Point& v);                   // copy constructor from a point 

	// Operations for computing length and normalizing
	inline float Normalize();
    inline float Length( void ) const;
	inline float LengthSqr( void ) const;

	// Assignment constructors
    inline Vector& operator=(const Vector& v);
    Vector& operator=(const Point& p);

	// Boolean comparisons
    inline bool operator==(const Vector& v) const;
	inline bool operator != (const Vector& v) const;
    
	// Mathematical operators
	inline Vector operator*(float s) const;
	friend inline Vector operator*(float s, const Vector& v);
    inline Vector operator*(const Vector& v) const;
    inline Vector operator/(const Vector& v) const;
	// inline Vector operator/(float s) const;  // This isn't defined, because it'll be inefficient!
    inline Vector& operator*=(float s);
    inline Vector operator+(const Vector& v) const;
    inline Vector& operator+=(const Vector& v);
	Vector operator+(const Point& p) const;
    inline Vector operator-() const;
    inline Vector operator-(const Vector& v) const;
	Vector operator-(const Point& p) const;

	// Dot and cross products
    inline Vector Cross(const Vector& v) const;
    inline float Dot(const Vector& v) const;
    float Dot(const Point& p) const;

	// Accessor functions
    inline float X() const;
    inline float Y() const;
    inline float Z() const;
	inline float W() const                 { return 0;}
	inline float GetElement( int i ) const { return d[i]; }
	inline float *GetDataPtr()             { return d; }

	// Static methods.  Useful for defining commonly used vectors
	static Vector Zero( void )  { return Vector(0,0,0); }
	static Vector One( void )   { return Vector(1,1,1); }
	static Vector XAxis( void ) { return Vector(1,0,0); }
	static Vector YAxis( void ) { return Vector(0,1,0); }
	static Vector ZAxis( void ) { return Vector(0,0,1); }

	// Performs component-wise max and min operations on vectors
	//    (NOTE: These are not particularly fast!)
	friend Vector Min(const Vector& v1, const Vector& v2);
    friend Vector Max(const Vector& v1, const Vector& v2);

	// Computes a scalar triple product
	friend float ScalarTripleProduct(const Vector& v1, const Vector& v2, const Vector &v3);

	// Returns the maximum (or minimum component)
	inline float MaxComponent( void ) const;
	inline float MinComponent( void ) const;

	inline void Print( void ) const { printf( "Vec: %f %f %f\n", d[0], d[1], d[2] ); }
};



inline Vector::Vector(float data[3])
{
    d[0]=data[0];
    d[1]=data[1];
    d[2]=data[2];
	d[3]=0;
}

inline Vector::Vector(float x, float y, float z) 
{
    d[0]=x;
    d[1]=y;
    d[2]=z;
	d[3]=0;
}

inline Vector::Vector(const Vector& v) 
{
    d[0]=v.d[0];
    d[1]=v.d[1];
    d[2]=v.d[2];
	d[3]=0;
}

inline float Vector::Length( void ) const 
{
    return sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
}

inline float Vector::LengthSqr( void ) const 
{
    return d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
}

inline Vector& Vector::operator=(const Vector& v) 
{
    d[0]=v.d[0];
    d[1]=v.d[1];
    d[2]=v.d[2];
    return *this;
}

inline Vector Vector::operator*(float s) const 
{
    return Vector(d[0]*s, d[1]*s, d[2]*s);
}

inline Vector operator*(float s, const Vector& v) 
{
	return Vector(v.d[0]*s, v.d[1]*s, v.d[2]*s);
}


inline Vector Vector::operator*(const Vector& v) const 
{
    return Vector(d[0]*v.d[0], d[1]*v.d[1], d[2]*v.d[2]);
}

inline Vector Vector::operator/(const Vector& v) const 
{
    return Vector(d[0]/v.d[0], d[1]/v.d[1], d[2]/v.d[2]);
}

inline Vector Vector::operator+(const Vector& v) const 
{
    return Vector(d[0]+v.d[0], d[1]+v.d[1], d[2]+v.d[2]);
}

inline Vector& Vector::operator+=(const Vector& v) 
{
    d[0]=d[0]+v.d[0];
    d[1]=d[1]+v.d[1];
    d[2]=d[2]+v.d[2];
    return *this;
}

inline Vector& Vector::operator*=(float s) 
{
    d[0]=d[0]*s;
    d[1]=d[1]*s;
    d[2]=d[2]*s;
    return *this;
}

inline Vector Vector::operator-() const 
{
    return Vector(-d[0], -d[1], -d[2]);
}

inline Vector Vector::operator-(const Vector& v) const 
{
    return Vector(d[0]-v.d[0], d[1]-v.d[1], d[2]-v.d[2]);
}

inline float Vector::Normalize() 
{
    float l=sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    d[0]=d[0]/l;
    d[1]=d[1]/l;
    d[2]=d[2]/l;
	//float l = ApproxInvSqrt( d[0]*d[0]+d[1]*d[1]+d[2]*d[2] );
	//d[0] = d[0]*l;
	//d[1] = d[1]*l;
	//d[2] = d[2]*l;
    return l;
}

inline Vector Vector::Cross(const Vector& v) const 
{
    return Vector(d[1]*v.d[2]-d[2]*v.d[1],
    	      d[2]*v.d[0]-d[0]*v.d[2],
    	      d[0]*v.d[1]-d[1]*v.d[0]);
}

inline float Vector::Dot(const Vector& v) const 
{
    return d[0]*v.d[0]+d[1]*v.d[1]+d[2]*v.d[2];
}

inline float Vector::X() const 
{
    return d[0];
}

inline float Vector::Y() const 
{
    return d[1];
}

inline float Vector::Z() const 
{
    return d[2];
}


inline bool Vector::operator != (const Vector& v) const {
    return d[0] != v.d[0] || d[1] != v.d[1] || d[2] != v.d[2];
}

inline bool Vector::operator == (const Vector& v) const {
    return d[0] == v.d[0] && d[1] == v.d[1] && d[2] == v.d[2];
}

inline float Vector::MinComponent( void ) const 
{
    float temp = (d[1] < d[0]? d[1] : d[0]); 
	return (d[2] < temp? d[2] : temp); 
}

inline float Vector::MaxComponent( void ) const 
{
	float temp = (d[1] > d[0]? d[1] : d[0]); 
	return (d[2] > temp? d[2] : temp); 
}


inline Vector Min(const Vector& v1, const Vector& v2)
{
    return Vector( MIN(v1.d[0], v2.d[0]), MIN(v1.d[1], v2.d[1]), MIN(v1.d[2], v2.d[2]) );
}

inline Vector Max(const Vector& v1, const Vector& v2)
{
    return Vector( MAX(v1.d[0], v2.d[0]), MAX(v1.d[1], v2.d[1]), MAX(v1.d[2], v2.d[2]) );
}

inline float ScalarTripleProduct(const Vector& v1, const Vector& v2, const Vector &v3)
{
	return v1.Dot( v2.Cross( v3 ) );
}

// Read a vector from a string
inline Vector::Vector( char *buffer )
{
	// Default values, in case something goes wrong.
	d[0] = d[1] = d[2] = 0; d[3]=0;

	char *ptr;
	ptr = StripLeadingNumber( buffer, &d[0] );
	ptr = StripLeadingNumber( ptr, &d[1] );
	ptr = StripLeadingNumber( ptr, &d[2] );
}


#endif

