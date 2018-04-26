


#ifndef __GLLAMBERTIANMATERIAL_H
#define __GLLAMBERTIANMATERIAL_H

#include "GLMaterial.h"

#pragma warning( disable: 4996 )

class GLLambertianMaterial : public GLMaterial
{
public:
	GLLambertianMaterial( char *matlName="<Unnamed Lambertian Material>" );
	GLLambertianMaterial( float *dif, char *matlName="<Unnamed Lambertian Material>" );
	GLLambertianMaterial( const Color &dif, char *matlName="<Unnamed Lambertian Material>" );
	GLLambertianMaterial( FILE *f, Scene *s );
	~GLLambertianMaterial() {}

	// Required material calls to enable and disable the material
	virtual void Enable( Scene *s, unsigned int flags=MATL_FLAGS_NONE );
	virtual void Disable( void )                    {}

	// Information about this type of material
	virtual bool UsesAlpha( void )					{ return (diffuse.Alpha()<1.0f); }
	virtual bool UsesLighting( void )				{ return true; }
};


#endif

