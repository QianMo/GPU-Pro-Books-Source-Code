/******************************************************************/
/* Bounce.h                                                       */
/* -----------------------                                        */
/*                                                                */
/* Implements a simple bounce animation routine for scene updates */
/*                                                                */
/* Chris Wyman (6/30/2008)                                        */
/******************************************************************/


#ifndef __BOUNCE_H__
#define __BOUNCE_H__

#include "ObjectTransform.h"

class Bounce : public ObjectTransform {
private:
	Vector dir;
	float speed;
	float start;
public:
	// Constructor & Destructor
	Bounce( FILE *f );
	virtual ~Bounce() {}

	// Applies the current transformation via glMultMatrix*()
	virtual void   Apply( void              );

	// Applies the current transformation to a vector
	virtual Vector Apply( const Vector &vec );

	// Applies the current transformation to a point
	virtual Point  Apply( const Point &pt   );

};



#endif