/******************************************************************/
/* Matrix4x4.h                                                    */
/* -----------------------                                        */
/*                                                                */
/* The file defines a basic matrix class that implments most of   */
/*     operations you need to perform on matrices, and interacts  */
/*     correctly with point and vector classes.                   */
/*                                                                */
/* Chris Wyman (02/23/2007)                                       */
/******************************************************************/

#ifndef MATRIX4X4_H
#define MATRIX4X4_H   1

#include <math.h>
#include <string.h>
#include <stdio.h>

#include "MathDefs.h"
#include "Utils/TextParsing.h"

class Point;
class Vector;

class Matrix4x4 {
    float mat[16]; 
    friend class Point; 
	friend class Vector;
public:
	// Matrix4x4 constructors
	inline Matrix4x4(float data[16]);         
    inline Matrix4x4(const Matrix4x4& v);       
	inline Matrix4x4(); 
	Matrix4x4( FILE *f, char *restOfLine=NULL );

	// Assignment constructors
    inline Matrix4x4& operator=(const Matrix4x4& v);

	// Accessor methods to access the matrix data
	inline float& operator[](const int index)              { return mat[index]; }
	inline float& operator()(const int col, const int row) { return mat[col*4+row]; }

	// Bad accessor method.  This breaks encapsulation, but is particularly useful in
	//    the context of OpenGL when passing matrices to the hardware.  Using this in
	//    other contexts is likely to 
	inline float *GetDataPtr()                             { return mat; }
    
	// Mathematical/assignment operators
    Matrix4x4& operator*=(const Matrix4x4& v);
	Matrix4x4& operator*=(const float s);
    Matrix4x4& operator+=(const Matrix4x4& v);
	Matrix4x4& operator-=(const Matrix4x4& v);

	// Mathematical matrix operators
	Matrix4x4 operator()(const Matrix4x4& v) const;
	Matrix4x4 operator*(const Matrix4x4& v)  const;
	Matrix4x4 operator*(const float s)       const;
    Matrix4x4 operator+(const Matrix4x4& v)  const;
	Matrix4x4 operator-(const Matrix4x4& v)  const;

	// Friend mathematical matrix operators
	friend Matrix4x4 operator*(const float s, const Matrix4x4& v);

	// This returns result^T, where result = v^T * m
	//     The result is a 3-component value (i.e., it ignores the 4th component!)
	friend Vector operator*(const Vector& v, const Matrix4x4& m);  

	// Mathematic point/vector operators
	Vector operator()(const Vector& v) const;
	Vector operator*(const Vector& v)  const;
	Point  operator()(const Point& v)  const;
	Point  operator*(const Point& v)   const;

	// Other interesting functions
	Matrix4x4 Invert( void ) const;
	Matrix4x4 Transpose( void ) const;
	float     Determinant( void ) const; // Not yet implemented!

	// Debug functions
	inline void Print( void );

	// Static constants
	static float identityArray[16];
	static float zeroArray[16];

	// Static methods.  Useful for defining commonly used matrices
	static Matrix4x4 Zero     ( void )          { return Matrix4x4( zeroArray );     }
	static Matrix4x4 Identity ( void )          { return Matrix4x4( identityArray ); }

	// Static methods for defining common transformations
	static Matrix4x4 Translate( float x, float y, float z );            // Not optimized for speed!
	static Matrix4x4 Scale    ( float x, float y, float z );            // Not optimized for speed!
	static Matrix4x4 Scale    ( float constantScale );                  // Not optimized for speed!
	static Matrix4x4 Rotate   ( float angle, const Vector& axis );      // Not optimized for speed!
	static Matrix4x4 Perspective  ( float fovy, float aspect, float zNear, float zFar );   // Not optimized for speed!
	static Matrix4x4 LookAt       ( const Point &eye, const Point &at, const Vector &up ); // Not optimized for speed!

	// Static methods for defining matrices given from outer products (of real-valued vectors)
	static Matrix4x4 OuterProduct ( const Vector& u );                  // Returns u * u^T
	static Matrix4x4 OuterProduct ( const Vector& u, const Vector& v ); // Returns u * v^T
};

inline void Matrix4x4::Print( void )
{
	printf("Matrix: %f %f %f %f\n", mat[0], mat[4], mat[8], mat[12] );
	printf("        %f %f %f %f\n", mat[1], mat[5], mat[9], mat[13] );
	printf("        %f %f %f %f\n", mat[2], mat[6], mat[10], mat[14] );
	printf("        %f %f %f %f\n", mat[3], mat[7], mat[11], mat[15] );
}

inline Matrix4x4::Matrix4x4(float data[16]) 
{
	memcpy( mat, data, 16*sizeof(float) );
}

inline Matrix4x4::Matrix4x4(const Matrix4x4& v)      
{
	memcpy( mat, v.mat, 16*sizeof(float) );
}

inline Matrix4x4::Matrix4x4()             
{
	memcpy( mat, Matrix4x4::identityArray, 16*sizeof(float) );
}

inline Matrix4x4& Matrix4x4::operator=(const Matrix4x4& v) 
{
	memcpy( mat, v.mat, 16*sizeof(float) );
    return *this;
}




#endif

