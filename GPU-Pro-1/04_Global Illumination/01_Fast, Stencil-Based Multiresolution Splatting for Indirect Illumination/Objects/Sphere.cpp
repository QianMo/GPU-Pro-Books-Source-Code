/******************************************************************/
/* Sphere.cpp                                                     */
/* -----------------------                                        */
/*                                                                */
/* The file defines a simple drawable OpenGL sphere type.         */
/*                                                                */
/* Chris Wyman (01/01/2008)                                       */
/******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "Utils/GLee.h"
#include <GL/glut.h>
#include "Sphere.h"
#include "Scene/Scene.h"
#include "Utils/ImageIO/imageIO.h"

Sphere::Sphere( Material *matl ) : 
	Object(matl), displayList(0), center( Point::Origin() ), 
	slices(20), stacks(20), radius(1.0f)
{	
}

void Sphere::Draw( Scene *s, unsigned int matlFlags, unsigned int optionFlags )
{
	if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
		matl->Enable( s, matlFlags );
	else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
		matl->EnableOnlyTextures( s );

	glCallList( displayList );

	if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
		matl->Disable();
	else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
		matl->DisableOnlyTextures();
}

// Draw this object (or it's sub-objects only if they have some property)
void Sphere::DrawOnly( Scene *s, unsigned int propertyFlags, unsigned int matlFlags, unsigned int optionFlags )
{
	if ((propertyFlags & flags) == propertyFlags) 
		this->Draw( s, matlFlags, optionFlags );
}


void Sphere::Preprocess( Scene *s )
{
	displayList = glGenLists(1);
	glNewList( displayList, GL_COMPILE );
	glPushMatrix();
	glTranslatef( center.X(), center.Y(), center.Z() );
	gluSphere( s->GetQuadric(), radius, slices, stacks );
	glPopMatrix();
	glEndList();
}


Sphere::Sphere( FILE *f, Scene *s ) :
	Object( s->GetDefaultMaterial() ),
	center( Point::Origin() ), slices(50), 
	stacks(25), displayList(0), radius(1.0f)
{
	bool normalsDefined=false;

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
		if (TestCommonObjectProperties( token, ptr, s, f ))
			continue;
		else if (!strcmp(token,"center"))
			center = Point( ptr );
		else if (!strcmp(token,"radius"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			radius = (float)atof( token );
		}
		else if (!strcmp(token, "stacks"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			stacks = (unsigned char)MIN(255,atoi( token ));
		}
		else if (!strcmp(token, "slices"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			slices = (unsigned char)MIN(255,atoi( token ));
		}
		else
			Error("Unknown command '%s' when loading Sphere!", token);
	}


}


