
#include <stdio.h>
#include <stdlib.h>
#include "GLMaterial.h"
#include "Utils/ImageIO/imageIO.h"
#include "Scene/Scene.h"

typedef struct {
  GLfloat ambient[4];
  GLfloat diffuse[4];
  GLfloat specular[4];
  GLfloat shiny;
} _mat_prop;


/* list of materials */
_mat_prop mat[19] = {
  { { 0.329412f, 0.223529f, 0.027451f, 1.0f},
    { 0.780392f, 0.568627f, 0.113725f, 1.0f},
    { 0.992157f, 0.941176f, 0.807843f, 1.0f},
    27.8974f
  },
  { { 0.2125f, 0.1275f, 0.054f, 1.0f},
    { 0.714f, 0.4284f, 0.18144f, 1.0f},
    { 0.393548f, 0.271906f, 0.166721f, 1.0f},
    25.6f
  },
  { { 0.25f, 0.148f, 0.06475f, 1.0f},
    { 0.4f, 0.2368f, 0.1036f, 1.0f},
    { 0.774597f, 0.458561f, 0.200621f, 1.0f},
    76.8f
  },
  { { 0.25f, 0.25f, 0.25f, 1.0f},
    { 0.4f, 0.4f, 0.4f, 1.0f},
    { 0.774597f, 0.774597f, 0.774597f, 1.0f},
    76.8f
  },
  { { 0.19125f, 0.0735f, 0.0225f, 1.0f},
    { 0.7038f, 0.27048f, 0.0828f, 1.0f},
    { 0.256777f, 0.137622f, 0.086014f, 1.0f},
    12.8f
  },
  { { 0.2295f, 0.08825f, 0.0275f, 1.0f},
    { 0.5508f, 0.2118f, 0.066f, 1.0f},
    { 0.580594f, 0.223257f, 0.0695701f, 1.0f},
    51.2f
  },
  { { 0.24725f, 0.1995f, 0.0745f, 1.0f},
    { 0.75164f, 0.60648f, 0.22648f, 1.0f},
    { 0.628281f, 0.555802f, 0.366065f, 1.0f},
    51.2f
  },
  { { 0.24725f, 0.2245f, 0.0645f, 1.0f},
    { 0.34615f, 0.3143f, 0.0903f, 1.0f},
    { 0.797357f, 0.723991f, 0.208006f, 1.0f},
    83.2f
  },
  { { 0.105882f, 0.058824f, 0.113725f, 1.0f},
    { 0.427451f, 0.470588f, 0.541176f, 1.0f},
    { 0.333333f, 0.333333f, 0.521569f, 1.0f},
    9.84615f
  },
  { { 0.19225f, 0.19225f, 0.19225f, 1.0f},
    { 0.50754f, 0.50754f, 0.50754f, 1.0f},
    { 0.508273f, 0.508273f, 0.508273f, 1.0f},
    51.2f
  },
  { { 0.23125f, 0.23125f, 0.23125f, 1.0f},
    { 0.2775f, 0.2775f, 0.2775f, 1.0f},
    { 0.773911f, 0.773911f, 0.773911f, 1.0f},
    89.6f
  },
  { { 0.0215f, 0.1745f, 0.0215f, 0.55f},
    { 0.07568f, 0.61424f, 0.07568f, 0.55f},
    { 0.633f, 0.727811f, 0.633f, 0.55f},
    76.8f
  },
  { { 0.135f, 0.2225f, 0.1575f, 0.95f},
    { 0.54f, 0.89f, 0.63f, 0.95f},
    { 0.316228f, 0.316228f, 0.316228f, 0.95f},
    12.8f
  },
  { { 0.05375f, 0.05f, 0.06625f, 0.82f},
    { 0.18275f, 0.17f, 0.22525f, 0.82f},
    { 0.332741f, 0.328634f, 0.346435f, 0.82f},
    38.4f
  },
  { { 0.25f, 0.20725f, 0.20725f, 0.922f},
    { 1.0f, 0.829f, 0.829f, 0.922f},
    { 0.296648f, 0.296648f, 0.296648f, 0.922f},
    11.264f
  },
  { { 0.1745f, 0.01175f, 0.01175f, 0.55f},
    { 0.61424f, 0.04136f, 0.04136f, 0.55f},
    { 0.727811f, 0.626959f, 0.626959f, 0.55f},
    76.8f
  },
  { { 0.1f, 0.18725f, 0.1745f, 0.8f},
    { 0.396f, 0.74151f, 0.69102f, 0.8f},
    { 0.297254f, 0.30829f, 0.306678f, 0.8f},
    12.8f
  },
  { { 0.0f, 0.0f, 0.0f, 1.0f},
    { 0.01f, 0.01f, 0.01f, 1.0f},
    { 0.50f, 0.50f, 0.50f, 1.0f},
    32.0f
  },
  { { 0.02f, 0.02f, 0.02f, 1.0f},
    { 0.01f, 0.01f, 0.01f, 1.0f},
    { 0.4f, 0.4f, 0.4f, 1.0f},
    10.0f
  }
};

char predefinedNames[19][80] = {
		  "MAT_BRASS" ,              
          "MAT_BRONZE",             
          "MAT_POLISHED_BRASS",     
          "MAT_CHROME",             
          "MAT_COPPER",             
          "MAT_POLISHED_COPPER",    
          "MAT_GOLD",               
          "MAT_POLISHED_GOLD",      
          "MAT_PEWTER",             
          "MAT_SILVER",             
          "MAT_POLISHED_SILVER",   
          "MAT_EMERALD",           
          "MAT_JADE",              
          "MAT_OBSIDIAN",          
          "MAT_PEARL",             
          "MAT_RUBY",              
          "MAT_TURQUOISE",         
          "MAT_BLACK_PLASTIC",     
          "MAT_BLACK_RUBBER"      
};


void GLMaterial::SetupShadowMap( GLenum texUnit, GLuint texID, float *matrix )
{
	glActiveTexture( texUnit );
	glBindTexture( GL_TEXTURE_2D, texID );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE );
	glTexGeni( GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR );
	glTexGeni( GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR );
	glTexGeni( GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR );
	glTexGeni( GL_Q, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR );
	glTexGenfv( GL_S, GL_EYE_PLANE, &matrix[0] );
	glTexGenfv( GL_T, GL_EYE_PLANE, &matrix[4] );
	glTexGenfv( GL_R, GL_EYE_PLANE, &matrix[8] );
	glTexGenfv( GL_Q, GL_EYE_PLANE, &matrix[12] );
	glEnable( GL_TEXTURE_2D );
	glEnable( GL_TEXTURE_GEN_S );
	glEnable( GL_TEXTURE_GEN_T );
	glEnable( GL_TEXTURE_GEN_R );
	glEnable( GL_TEXTURE_GEN_Q );
	glActiveTexture( GL_TEXTURE0 );
}

void GLMaterial::DisableShadowMap( GLenum texUnit )
{
	glActiveTexture( texUnit );
	glDisable( GL_TEXTURE_2D );
	glDisable( GL_TEXTURE_GEN_S );
	glDisable( GL_TEXTURE_GEN_T );
	glDisable( GL_TEXTURE_GEN_R );
	glDisable( GL_TEXTURE_GEN_Q );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE );
	glBindTexture( GL_TEXTURE_2D, 0 );
	glActiveTexture( GL_TEXTURE0 );
}


void GLMaterial::Enable( Scene *s, unsigned int flags )
{
	usingShadows = (flags & MATL_FLAGS_USESHADOWMAP); 

	glMaterialfv( whichFace, GL_AMBIENT, ambient.GetDataPtr() );
	glMaterialfv( whichFace, GL_DIFFUSE, diffuse.GetDataPtr() );
	glMaterialfv( whichFace, GL_SPECULAR, specular.GetDataPtr() );
	glMaterialfv( whichFace, GL_EMISSION, emission.GetDataPtr() );
	glMaterialf( whichFace, GL_SHININESS, shininess );
	if (usingShadows) 
	{   // Please NOTE.  This does not seem to work.  Blah.  Needs fixing.
		glPushMatrix();
		glLoadIdentity();
		s->LookAtMatrix();
		SetupShadowMap( GL_TEXTURE0, s->GetShadowMapID( 0 ), s->GetShadowMapTransposeMatrix( 0 ) );
		glPopMatrix();
	}
}

void GLMaterial::Disable( void )                    
{
	if (usingShadows) 
		DisableShadowMap( GL_TEXTURE0 );
}


void GLMaterial::EnableOnlyTextures( Scene *s )
{
}

void GLMaterial::DisableOnlyTextures( void )
{
}


GLMaterial::GLMaterial( int predefined ) : 
	Material(), whichFace(GL_FRONT_AND_BACK)
{
	SetName( predefinedNames[predefined] );

	ambient = Color( mat[predefined].ambient );
	diffuse = Color( mat[predefined].diffuse );
	specular = Color( mat[predefined].specular );
	emission = Color::Black();
	shininess = mat[predefined].shiny;
}

GLMaterial::GLMaterial( char *matlName ) : Material( matlName ),
	ambient( 0.2f, 0.2f, 0.2f ), diffuse( 0.8f, 0.8f, 0.8f ),
	specular( Color::Black() ), emission( Color::Black() ), 
	shininess( 65.0f ), whichFace(GL_FRONT_AND_BACK)
{
}

GLMaterial::GLMaterial( float *amb, float *dif, float *spec, float shiny, char *matlName ) :
	Material( matlName ), ambient( amb ), diffuse( dif ), specular( spec ), 
	shininess( shiny ), emission( Color::Black() ), whichFace(GL_FRONT_AND_BACK)
{
}

GLMaterial::GLMaterial( const Color &amb, const Color &dif, 
		        const Color &spec, float shiny, char *matlName ) :
	Material( matlName ), ambient( amb ), diffuse( dif ), specular( spec ), 
	shininess( shiny ), emission( Color::Black() ), whichFace(GL_FRONT_AND_BACK)
{
}


GLMaterial::GLMaterial( FILE *f, Scene *s ) : 
	Material(), ambient( 0.2f, 0.2f, 0.2f ), diffuse( 0.8f, 0.8f, 0.8f ),
	specular( Color::Black() ), emission( Color::Black() ), 
	shininess( 65.0f ), whichFace(GL_FRONT_AND_BACK)
{
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
		if (!strcmp(token,"ambient") || !strcmp(token,"amb")) 
			ambient = Color( ptr ); 
		else if (!strcmp(token,"diffuse") || !strcmp(token,"dif")) 
			diffuse = Color( ptr ); 
		else if (!strcmp(token,"specular") || !strcmp(token,"spec")) 
			specular = Color( ptr ); 
		else if (!strcmp(token,"emission") || !strcmp(token,"emit") )  
			emission = Color( ptr ); 
		else if (!strcmp(token,"shininess") || !strcmp(token,"shiny") )  
			StripLeadingNumber( ptr, &shininess );
		else if (!strcmp(token,"name"))  
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			SetName( token );
		}
		else
			Error("Unknown command '%s' when loading GLMaterial!", token);
	}

}

