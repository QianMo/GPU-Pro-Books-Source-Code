/******************************************************************/
/* Cylinder.h                                                       */
/* -----------------------                                        */
/*                                                                */
/* The file defines a simple drawable OpenGL sphere type.         */
/*                                                                */
/* Chris Wyman (01/01/2008)                                       */
/******************************************************************/


#ifndef __CYLINDER_H__
#define __CYLINDER_H__


#include "Object.h"
#include "DataTypes/Vector.h"
#include "DataTypes/Point.h"
#include "DataTypes/Color.h"


class Scene;

class Cylinder : public Object {
public:
	Cylinder( Material *matl=0 );
	Cylinder( FILE *f, Scene *s );
	virtual ~Cylinder() {}

	virtual void Draw( Scene *s, 
		               unsigned int matlFlags, 
					   unsigned int optionFlags=OBJECT_OPTION_NONE );
	virtual void DrawOnly( Scene *s, 
		                   unsigned int propertyFlags, 
						   unsigned int matlFlags, 
						   unsigned int optionFlags=OBJECT_OPTION_NONE );

	virtual bool NeedsPreprocessing( void ) { return displayList == 0; }
	virtual void Preprocess( Scene *s );

private:
	Vector axis;
	GLuint displayList;
	float radius, height;
	Point center;
	unsigned char stacks, slices;
};

#endif

