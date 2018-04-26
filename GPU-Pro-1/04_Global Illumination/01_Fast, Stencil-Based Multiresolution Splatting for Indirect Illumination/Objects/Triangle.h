/******************************************************************/
/* Triangle.h                                                     */
/* -----------------------                                        */
/*                                                                */
/* The file defines a simple drawable OpenGL triangle type.       */
/*                                                                */
/* Chris Wyman (01/01/2008)                                       */
/******************************************************************/

#include "Object.h"
#include "DataTypes/Vector.h"
#include "DataTypes/Point.h"
#include "DataTypes/Color.h"

#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

class Triangle : public Object {
public:
	Triangle( Material *matl=0 ) : Object(matl) {}
	Triangle( FILE *f, Scene *s );
	virtual ~Triangle() {}

	virtual void Draw( Scene *s, 
		               unsigned int matlFlags, 
					   unsigned int optionFlags=OBJECT_FLAGS_NONE );
	virtual void DrawOnly( Scene *s, 
		                   unsigned int propertyFlags, 
						   unsigned int matlFlags, 
						   unsigned int optionFlags=OBJECT_FLAGS_NONE );

private:
	Point vert0, vert1, vert2;
	Vector tex0, tex1, tex2;
	Vector norm0, norm1, norm2;
};

#endif

