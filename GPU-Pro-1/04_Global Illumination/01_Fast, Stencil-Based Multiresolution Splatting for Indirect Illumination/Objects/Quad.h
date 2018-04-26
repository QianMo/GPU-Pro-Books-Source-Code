/******************************************************************/
/* Quad.h                                                         */
/* -----------------------                                        */
/*                                                                */
/* The file defines a simple drawable OpenGL quad type.           */
/*                                                                */
/* Chris Wyman (01/01/2008)                                       */
/******************************************************************/

#include "Object.h"
#include "DataTypes/Vector.h"
#include "DataTypes/Point.h"
#include "DataTypes/Color.h"

#ifndef __QUAD_H__
#define __QUAD_H__

class Quad : public Object {
public:
	Quad( Material *matl=0 ) : Object(matl) {}
	Quad( FILE *f, Scene *s );
	virtual ~Quad() {}

	virtual void Draw( Scene *s, 
		               unsigned int matlFlags, 
					   unsigned int optionFlags=OBJECT_OPTION_NONE );
	virtual void DrawOnly( Scene *s, 
		                   unsigned int propertyFlags, 
						   unsigned int matlFlags, 
						   unsigned int optionFlags=OBJECT_OPTION_NONE );

private:
	Point vert0, vert1, vert2, vert3;
	Vector tex0, tex1, tex2, tex3;
	Vector norm0, norm1, norm2, norm3;
};

#endif


