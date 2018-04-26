


#ifndef __GLLAMBERTIANTEXMATERIAL_H
#define __GLLAMBERTIANTEXMATERIAL_H

#include "GLMaterial.h"
#include "DataTypes/glTexture.h"

#pragma warning( disable: 4996 )

class GLLambertianTexMaterial : public GLMaterial
{
private:
	GLTexture *tex;
public:
	GLLambertianTexMaterial( char *matlName="<Unnamed Lambertian Material>" );
	GLLambertianTexMaterial( float *dif, char *matlName="<Unnamed Lambertian Material>" );
	GLLambertianTexMaterial( const Color &dif, char *matlName="<Unnamed Lambertian Material>" );
	GLLambertianTexMaterial( FILE *f, Scene *s );
	~GLLambertianTexMaterial() {}

	// Required material calls to enable and disable the material
	virtual void Enable( Scene *s, unsigned int flags=MATL_FLAGS_NONE );
	virtual void Disable( void );

	virtual void EnableOnlyTextures( Scene *s );
	virtual void DisableOnlyTextures( void );

	// Information about this type of material
	virtual bool UsesAlpha( void )					{ return (diffuse.Alpha()<1.0f); }
	virtual bool UsesLighting( void )				{ return true; }
	virtual bool UsesTexture( void )				{ return tex!=0; }

	// Allow queries to get the material's texture
	virtual GLTexture *GetMaterialTexture( void )	{ return tex; }
};


#endif

