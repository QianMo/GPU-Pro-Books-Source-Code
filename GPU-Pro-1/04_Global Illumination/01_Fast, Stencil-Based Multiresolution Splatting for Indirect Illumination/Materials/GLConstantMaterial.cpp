
#include <stdio.h>
#include <stdlib.h>
#include "Scene/Scene.h"
#include "GLConstantMaterial.h"
#include "Utils/ImageIO/imageIO.h"


void GLConstantMaterial::Enable( Scene *s, unsigned int flags )
{
	glMaterialfv( whichFace, GL_AMBIENT, ambient.GetDataPtr() );
	glMaterialfv( whichFace, GL_DIFFUSE, diffuse.GetDataPtr() );
	glMaterialfv( whichFace, GL_SPECULAR, specular.GetDataPtr() );
	glMaterialfv( whichFace, GL_EMISSION, emission.GetDataPtr() );
	glMaterialf( whichFace, GL_SHININESS, shininess );
}


GLConstantMaterial::GLConstantMaterial( char *matlName ) : GLMaterial( matlName )
{
	ambient = Color::Black();
	diffuse = Color::Black();
	specular = Color::Black();
	emission = Color( 0.8f, 0.8f, 0.8f );
	shininess = 0.0f; 
}

GLConstantMaterial::GLConstantMaterial( float *color, char *matlName ) :
	GLMaterial( matlName )
{
	ambient = Color::Black();
	diffuse = Color::Black();
	specular = Color::Black();
	emission = Color( color );
	shininess = 0.0f; 
}

GLConstantMaterial::GLConstantMaterial( const Color &color, char *matlName ) :
	GLMaterial( matlName )
{
	ambient = Color::Black();
	diffuse = Color::Black();
	specular = Color::Black();
	emission = color;
	shininess = 0.0f; 
}


GLConstantMaterial::GLConstantMaterial( FILE *f, Scene *s ) : 
	GLMaterial()
{
	ambient = Color::Black();
	diffuse = Color::Black();
	specular = Color::Black();
	emission = Color( 0.8f, 0.8f, 0.8f );
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
		if (!strcmp(token,"rgb") || !strcmp(token,"color") )
			emission = Color( ptr );
		else if (!strcmp(token,"name"))  
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			SetName( token );
		}
		else if (!strcmp(token,"tex") || !strcmp(token,"texture") || 
				 !strcmp(token,"clamp") || !strcmp(token,"mirror") || !strcmp(token,"repeat"))
			 s->UnhandledKeyword( f, "constant material property", token );
		else
			Error("Unknown command '%s' when loading GLConstantMaterial!", token);
	}

}

