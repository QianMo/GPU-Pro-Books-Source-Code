


#ifndef __GLCONSTANTMATERIAL_H
#define __GLCONSTANTMATERIAL_H

#include "GLMaterial.h"

#pragma warning( disable: 4996 )

class GLConstantMaterial : public GLMaterial
{
public:
	GLConstantMaterial( char *matlName="<Unnamed Constant Material>" );
	GLConstantMaterial( float *color, char *matlName="<Unnamed Constant Material>" );
	GLConstantMaterial( const Color &color, char *matlName="<Unnamed Constant Material>" );
	GLConstantMaterial( FILE *f, Scene *s );
	~GLConstantMaterial() {}

	// Required material calls to enable and disable the material
	virtual void Enable( Scene *s, unsigned int flags=MATL_FLAGS_NONE );
	virtual void Disable( void )                    {}

	// Information about this type of material
	virtual bool UsesAlpha( void )					{ return false; }
	virtual bool UsesLighting( void )				{ return true; }
};


#endif

