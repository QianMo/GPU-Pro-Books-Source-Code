#ifndef TRACKBALL_H
#define TRACKBALL_H

#include <stdio.h>
#include <stdlib.h>
#include "Utils/GLee.h"
#include <GL/glut.h>
#include "DataTypes/Matrix4x4.h"
#include "DataTypes/Vector.h"
#include "DataTypes/Point.h"

class Trackball;

class TrackballList
{
private:
	Trackball **list;
	int listSize;

public:
	/* sets up a trackball list of trackballs with the same width and height */
	TrackballList( int num, int width, int height );
	~TrackballList();

	/* returns a pointer to a trackball */
	Trackball *GetBall( int i ) { return ( i>=0 && i<listSize ? list[i] : NULL); }

	/* goes thru the list calling list[i]->ResizeTrackball( w, h ) */
	void ResizeTrackballs( int width, int height );
};


class Trackball 
{
private:
	int hasChanged;
	int currentlyTracking;
	int ballWidth, ballHeight;
	float lastPos[3]; 
	Matrix4x4 trackballMatrix;
	Matrix4x4 inverseTrackballMatrix;
	//float trackballMatrix[16]; 
	//float inverseTrackballMatrix[16]; 

	void trackball_ptov(int x, int y, int width, int height, float v[3]);

public:

	/* sets up a trackball with window size of width x height */
	Trackball( int width, int height );
	~Trackball();

	/* functions for updating the trackball due to mouse input.           */
	/*    x & y are the window coordinates of the current mouse position. */
	void SetTrackballOnClick( int x, int y );
	void UpdateTrackballOnMotion( int x, int y );
	inline void ReleaseTrackball( void ) { currentlyTracking = 0; }

	/* checks if the trackball has been updated since the most recent of:  */
	/*   a) trackball initialization or b) the last call to HasChanged()   */
	/*   Please note: HasChanged() destructively checks the internal value */
	/*   (i.e., it sets the internal hasChanged variable to false!)        */
	bool HasChanged( void );

	/* the computations rely on a width and height of the window (for expected operation) */
	void ResizeTrackballWindow( int width, int height );

	/* calls to multiply the trackball's matrix onto the current GL stack.   */
	void MultiplyTrackballMatrix( void );
	void MultiplyInverseTrackballMatrix( void );
	void MultiplyTransposeTrackballMatrix( void );
	void MultiplyInverseTransposeTrackballMatrix( void );

	/* Apply the current trackball matrix to a 4-vector */
	void ApplyTrackballMatrix( float inVec[4], float result[4] );
	Vector ApplyTrackballMatrix( const Vector &vec );
	Point ApplyTrackballMatrix( const Point &pt );

	/* calls to get or set the value of the trackball */
	void GetTrackBallMatrix( float *matrix );
	inline const Matrix4x4 &GetTrackBallMatrix( void )		{ return trackballMatrix; }
	void SetTrackballMatrix( float *newMat );
	void SetTrackballMatrix( const Matrix4x4 &newMat );

	/* Gets the inverse of the trackball matrix */
	void GetInverseTrackBallMatrix( GLfloat *matrix );
	inline const Matrix4x4 &GetInverseTrackBallMatrix( void )		{ return inverseTrackballMatrix; }

	/* call to print to standard out the matrix values. */
	void PrintTrackballMatrix( void );

	/* if the trackball needs to be reset... */
	void ResetTrackball( void );
};



#endif
