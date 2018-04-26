/******************************************************************/
/* Rotate.h                                                       */
/* -----------------------                                        */
/*                                                                */
/* Implements a simple rotation animation routine for obj updates */
/*                                                                */
/* Chris Wyman (6/30/2008)                                        */
/******************************************************************/


#ifndef __ROTATE_H__
#define __ROTATE_H__

#include "ObjectTransform.h"

class Rotate : public ObjectTransform {
private:
	Vector axis;
	float speed;
	float start;
public:
	// Constructor & Destructor
	Rotate( FILE *f );
	virtual ~Rotate() {}

	// Applies the current transformation via glMultMatrix*()
	virtual void   Apply( void              );

	// Applies the current transformation to a vector
	virtual Vector Apply( const Vector &vec );

	// Applies the current transformation to a point
	virtual Point  Apply( const Point &pt   );
};



#endif