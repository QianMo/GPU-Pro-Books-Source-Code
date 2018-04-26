#ifndef GLMATERIAL_H
#define GLMATERIAL_H

#include "Utils/GLee.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DataTypes/Color.h"
#include "Material.h"
#include "Utils/TextParsing.h"

#pragma warning( disable: 4996 )

class GLMaterial : public Material
{
protected:
	// basic material properties
	Color ambient, diffuse, specular, emission;
	float shininess;
	GLenum whichFace;

	bool usingShadows;

	void SetupShadowMap( GLenum texUnit, GLuint texID, float *matrix );
	void DisableShadowMap( GLenum texUnit );

public:
	GLMaterial( int predefined );
	GLMaterial( char *matlName="<Unnamed Material>" );
	GLMaterial( float *amb, float *dif, float *spec, float shiny, char *matlName="<Unnamed Material>" );
	GLMaterial( const Color &amb, const Color &dif, 
		        const Color &spec, float shiny, char *matlName="<Unnamed Material>" );
	GLMaterial( FILE *f, Scene *s );
	~GLMaterial() {}

	// Required material calls to enable and disable the material
	virtual void Enable( Scene *s, unsigned int flags=MATL_FLAGS_NONE );
	virtual void Disable( void );

	// Sometimes materials have an associated texture that we want bound
	//    _even_if_ we use a different shader / rendering approach.  This
	//    should not bind shadow maps or other lighting-based textures, just
	//    details such as normal maps and basic texture maps.  Presumably,
	//    for this to make any sense, all the materials in your scene must
	//    use consistant binding locations (e.g., GL_TEXTURE1 for color tex)
	virtual void EnableOnlyTextures( Scene *s );
	virtual void DisableOnlyTextures( void );

	// Set material parameters 
	inline void SetAmbient( const Color &amb )		{ ambient = amb; }
	inline void SetDiffuse( const Color &dif )		{ diffuse = dif; }
	inline void SetSpecular( const Color &spec )	{ specular = spec; }
	inline void SetEmission( const Color &emit )	{ emission = emit; }
	inline void SetShininess( float shiny )			{ shininess = shiny; }

	// Get material parameters 
	inline const Color &GetAmbient( void )			{ return ambient; }
	inline const Color &GetDiffuse( void )			{ return diffuse; }
	inline const Color &GetSpecular( void )			{ return specular; }
	inline const Color &GetEmission( void )			{ return emission; }
	inline float GetShininess( void ) const			{ return shininess; }

	// Information about this type of material
	virtual bool UsesAlpha( void )					{ return (ambient.Alpha()<1.0f) || (diffuse.Alpha()<1.0f) || (specular.Alpha()<1.0f); }
	virtual bool UsesLighting( void )				{ return true; }
};




/* Predefined Material Types */
#define MAT_BRASS              0
#define MAT_BRONZE             1
#define MAT_POLISHED_BRASS     2
#define MAT_CHROME             3
#define MAT_COPPER             4
#define MAT_POLISHED_COPPER    5
#define MAT_GOLD               6
#define MAT_POLISHED_GOLD      7
#define MAT_PEWTER             8
#define MAT_SILVER             9
#define MAT_POLISHED_SILVER   10
#define MAT_EMERALD           11
#define MAT_JADE              12
#define MAT_OBSIDIAN          13
#define MAT_PEARL             14
#define MAT_RUBY              15
#define MAT_TURQUOISE         16
#define MAT_BLACK_PLASTIC     17
#define MAT_BLACK_RUBBER      18


#endif

