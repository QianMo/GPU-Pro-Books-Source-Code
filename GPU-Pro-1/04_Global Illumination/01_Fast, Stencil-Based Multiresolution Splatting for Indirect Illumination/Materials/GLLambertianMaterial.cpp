
#include <stdio.h>
#include <stdlib.h>
#include "Scene/Scene.h"
#include "GLLambertianMaterial.h"
#include "Utils/ImageIO/imageIO.h"


void GLLambertianMaterial::Enable( Scene *s, unsigned int flags )
{
	glMaterialfv( whichFace, GL_AMBIENT, ambient.GetDataPtr() );
	glMaterialfv( whichFace, GL_DIFFUSE, diffuse.GetDataPtr() );
	glMaterialfv( whichFace, GL_SPECULAR, specular.GetDataPtr() );
	glMaterialfv( whichFace, GL_EMISSION, emission.GetDataPtr() );
	glMaterialf( whichFace, GL_SHININESS, shininess );
}


GLLambertianMaterial::GLLambertianMaterial( char *matlName ) : GLMaterial( matlName )
{
	ambient = Color::Black();
	diffuse = Color( 0.8f, 0.8f, 0.8f );
	specular = Color::Black();
	emission = Color::Black();
	shininess = 0.0f; 
}

GLLambertianMaterial::GLLambertianMaterial( float *dif, char *matlName ) :
	GLMaterial( matlName )
{
	ambient = Color::Black();
	diffuse = Color( dif );
	specular = Color::Black();
	emission = Color::Black();
	shininess = 0.0f; 
}

GLLambertianMaterial::GLLambertianMaterial( const Color &dif, char *matlName ) :
	GLMaterial( matlName )
{
	ambient = Color::Black();
	diffuse = dif;
	specular = Color::Black();
	emission = Color::Black();
	shininess = 0.0f; 
}


GLLambertianMaterial::GLLambertianMaterial( FILE *f, Scene *s ) : 
	GLMaterial()
{
	ambient = Color::Black();
	diffuse = Color( 0.8f, 0.8f, 0.8f );
	specular = Color::Black();
	emission = Color::Black();
	shininess = 0.0f; 

	// Search the scene file.
	char buf[ MAXLINELENGTH ], token[256], *ptr;
	while( fgets(buf, MAXLINELENGTH, f) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr[0] == '#') continue;

		// Nope.  So find out what the command is...
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
	
		// Take different measures, depending on the command.
		if (!strcmp(token,"end")) break;
		if (!strcmp(token,"rgb") || 
			!strcmp(token,"color") || !strcmp(token,"albedo"))
			diffuse = Color( ptr );
		else if (!strcmp(token,"spectral"))
			FatalError("Unable to handle keyword 'spectral' in GLLambertianMaterial!");
		else if (!strcmp(token,"name"))  
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			SetName( token );
		}
		else if (!strcmp(token,"tex") || !strcmp(token,"texture") || 
				 !strcmp(token,"diffusetexture") || !strcmp(token,"dtex") ||
				 !strcmp(token,"clamp") || !strcmp(token,"mirror") || !strcmp(token,"repeat"))
			 s->UnhandledKeyword( f, "lambertian material property", token );
		else
			Error("Unknown command '%s' when loading GLMaterial!", token);
	}

}

