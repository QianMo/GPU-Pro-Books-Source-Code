
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Scene/Scene.h"
#include "GLLambertianTexMaterial.h"
#include "DataTypes/glTexture.h"
#include "Utils/ImageIO/imageIO.h"


void GLLambertianTexMaterial::Enable( Scene *s, unsigned int flags )
{
	glMaterialfv( whichFace, GL_AMBIENT, ambient.GetDataPtr() );
	glMaterialfv( whichFace, GL_DIFFUSE, diffuse.GetDataPtr() );
	glMaterialfv( whichFace, GL_SPECULAR, specular.GetDataPtr() );
	glMaterialfv( whichFace, GL_EMISSION, emission.GetDataPtr() );
	glMaterialf( whichFace, GL_SHININESS, shininess );
	if (tex)
	{
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, tex->TextureID() );
		glEnable( GL_TEXTURE_2D );
	}
}

void GLLambertianTexMaterial::Disable( void )
{
	if (tex)
	{
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, 0 );
		glDisable( GL_TEXTURE_2D );
	}
}

void GLLambertianTexMaterial::EnableOnlyTextures( Scene *s )
{
	if (tex)
	{
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, tex->TextureID() );
		glEnable( GL_TEXTURE_2D );
	}
}

void GLLambertianTexMaterial::DisableOnlyTextures( void )
{
	if (tex)
	{
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, 0 );
		glDisable( GL_TEXTURE_2D );
	}
}


GLLambertianTexMaterial::GLLambertianTexMaterial( char *matlName ) : GLMaterial( matlName ), tex(0)
{
	ambient = Color::Black();
	diffuse = Color( 0.8f, 0.8f, 0.8f );
	specular = Color::Black();
	emission = Color::Black();
	shininess = 0.0f; 
}

GLLambertianTexMaterial::GLLambertianTexMaterial( float *dif, char *matlName ) :
	GLMaterial( matlName ), tex(0)
{
	ambient = Color::Black();
	diffuse = Color( dif );
	specular = Color::Black();
	emission = Color::Black();
	shininess = 0.0f; 
}

GLLambertianTexMaterial::GLLambertianTexMaterial( const Color &dif, char *matlName ) :
	GLMaterial( matlName ), tex(0)
{
	ambient = Color::Black();
	diffuse = dif;
	specular = Color::Black();
	emission = Color::Black();
	shininess = 0.0f; 
}


GLLambertianTexMaterial::GLLambertianTexMaterial( FILE *f, Scene *s ) : 
	GLMaterial(), tex(0)
{
	char filename[257];
	ambient = Color::Black();
	diffuse = Color( 0.8f, 0.8f, 0.8f );
	specular = Color::Black();
	emission = Color::Black();
	shininess = 0.0f; 
	filename[0] = 0;
	unsigned int flags = TEXTURE_REPEAT_S | TEXTURE_REPEAT_T | TEXTURE_MIN_LINEAR_MIP_LINEAR;

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
			FatalError("Unable to handle keyword 'spectral' in GLLambertianTexMaterial!");
		else if (!strcmp(token,"name"))  
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			SetName( token );
		}
		else if (!strcmp(token,"tex") || !strcmp(token,"texture") || 
				 !strcmp(token,"diffusetexture") || !strcmp(token,"dtex"))
				 
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			strcpy(filename,token);
		}
		else if (!strcmp(token,"clamp"))
			flags |= TEXTURE_CLAMP_S | TEXTURE_CLAMP_T;
		else if (!strcmp(token,"mirror"))
			flags |= TEXTURE_MIRROR_REPEAT_S | TEXTURE_MIRROR_REPEAT_T;
		else if (!strcmp(token,"repeat"))
			flags |= TEXTURE_REPEAT_S | TEXTURE_REPEAT_T;
		else
			Error("Unknown command '%s' when loading GLMaterial!", token);
	}

	if (filename[0] != 0)
	{
		tex = s->GetNamedTexture( filename );
		if (!tex)
		{
			char *file = s->paths->GetTexturePath( filename );
			if (file)
			{
				tex = s->ExistingTextureFromFile( file );
				if (!tex)
					s->AddTexture( tex = new GLTexture( file, flags, true ));
				free( file );
			}
		}
	}


}

