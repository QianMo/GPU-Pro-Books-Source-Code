/************************************************************************/
/* areaLight.h                                                          */
/* ------------------                                                   */
/*                                                                      */
/* This file defines a class storing information about an area light,   */
/*     as well as access routines.                                      */
/*                                                                      */
/* Chris Wyman (05/12/2009)                                             */
/************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "areaLight.h"
#include "Scene/Scene.h"
#include "Utils/Trackball.h"
#include "Utils/TextParsing.h"
#include "DataTypes/Matrix4x4.h"
#include "DataTypes/Point.h"
#include "DataTypes/Vector.h"
#include "DataTypes/Color.h"
#include "Utils/glslProgram.h"
#include "DataTypes/glTexture.h"
#include "DataTypes/glVideoTexture.h"


AreaLight::AreaLight( FILE *f, Scene *s ) :
		lightColor( 1, 1, 1 ), lightTex( NULL ),
		pos( 0, 0, 0 ), edge1( 1, 0, 0 ), edge2( 0, 1, 0 ),
		name(0)
{ 
	// Search the scene file.
	int flag=0;
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
		if ( !strcmp(token,"rgb") || !strcmp(token,"color"))         // Load the light's color    
			lightColor = LoadColor::Load( token, f, ptr );
		else if (!strcmp(token,"pos") || !strcmp(token,"position") ) // Load corner of area light
			pos = Point( ptr );
		else if (!strcmp(token,"e1") || !strcmp(token,"edge1") )     // Load 1st edge of area light
			edge1 = Vector( ptr );
		else if (!strcmp(token,"e2") || !strcmp(token,"edge2") )     // Load 2nd edge of area light
			edge2 = Vector( ptr );
		else if (!strcmp(token,"name"))                              // Load arbitrary name
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			this->SetAreaLightName( token );
		}
		else if ( (flag = !strcmp(token, "tex")) || !strcmp(token, "videotex") ) // Load texture on the light
			{ // Bind a texture
				ptr = StripLeadingTokenToBuffer( ptr, token );
				GLTexture *tptr = s->GetNamedTexture( token );       // Check: Is token a texture name?
				char *texFile=0;
				if (!tptr)                                           // No.  Check: Is a filename that  
				{													 //    we already loaded as a texture?
					texFile = s->paths->GetTexturePath( token );
					tptr = s->ExistingTextureFromFile( texFile );
				}
				if (!tptr)                                           // No.  New texture.  Load it.
				{
					// Different type to load if this was a static texture or video texture
					if (flag)
						tptr = new GLTexture( texFile, 
									   TEXTURE_REPEAT_S | TEXTURE_REPEAT_T | TEXTURE_MIN_LINEAR_MIP_LINEAR, 
									   true );
					else
						tptr = new GLVideoTexture( texFile, 30.0, 
									   TEXTURE_REPEAT_S | TEXTURE_REPEAT_T | TEXTURE_MIN_LINEAR_MIP_LINEAR );
					s->AddTexture( tptr );
				}
			
				// If we allocated a path name, free that.
				if (texFile) free( texFile );

				// If we have a old texture (and a valid new one), delete old texture
				if (lightTex && tptr) delete lightTex;
				lightTex = tptr;
			}
		else
			printf("Error: Unknown command '%s' when loading Light\n", token);
	}

	// Figure out the surface normal
	surfNorm = edge1.Cross( edge2 );
	surfNorm.Normalize();
}


AreaLight::~AreaLight()
{
	if ( name ) free( name );
	if ( lightTex ) delete lightTex;
}


bool AreaLight::NeedPerFrameUpdates( void )
{
	if (lightTex && lightTex->NeedsUpdates()) 
		return true;
	return false;
}


void AreaLight::Update( float currentTime )
{
	if (lightTex)
		lightTex->Update( currentTime );
}
