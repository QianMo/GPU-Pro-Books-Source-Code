/******************************************************************/
/* ObjectTransform.h                                              */
/* -----------------------                                        */
/*                                                                */
/* Base class for object transformations.  Object transformations */
/*    read their specifications from a file, then are updated     */
/*    each frame with a call to Update( currentTime ), and are    */
/*    utilized by the program by calling Apply(), which should    */
/*    invoke glMultMatrix*() to multiply the current transform    */
/*    onto the current OpenGL matrix stack.                       */
/*                                                                */
/* Chris Wyman (6/30/2008)                                        */
/******************************************************************/


#ifndef __OBJECTTRANSFORM_H__
#define __OBJECTTRANSFORM_H__

#include <stdio.h>
#include "DataTypes/Point.h"
#include "DataTypes/Vector.h"
#include "DataTypes/Matrix4x4.h"

class ObjectTransform {
protected:
	float curTime;
public:
	// Constructor & Destructor
	ObjectTransform() : curTime(0) {}
	virtual ~ObjectTransform() {}

	// Updates the transformation based upon the current time
	virtual void Update( float currentTime ) { curTime = currentTime; }

	// Applies the current transformation via glMultMatrix*()
	virtual void   Apply( void              ) = 0;

	// Applies the current transformation to a vector
	virtual Vector Apply( const Vector &vec ) = 0;

	// Applies the current transformation to a point
	virtual Point  Apply( const Point &pt   ) = 0;
};


// Prototype for Error() used inside some of these object transforms
//    Prototype originally defined in ImageIO/ImageIO.h
void Error( char *formatStr, char *insertThisStr );

#endif