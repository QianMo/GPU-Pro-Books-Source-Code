/******************************************************************/
/* Bounce.cpp                                                     */
/* -----------------------                                        */
/*                                                                */
/* Implements a simple bounce animation routine for scene updates */
/*                                                                */
/* Chris Wyman (6/30/2008)                                        */
/******************************************************************/

#include <stdio.h>
#include <math.h>
#include "Bounce.h"
#include "Utils/GLee.h"
#include <GL/glut.h>


// Constructor & Destructor
Bounce::Bounce( FILE *f ) : 
	start(0.0), speed(1.0), dir(0,1,0)
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
		if (!strcmp(token,"dir") || !strcmp(token, "direction") || !strcmp(token, "vector"))
			dir = Vector( ptr );
		else if (!strcmp(token,"speed"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			speed = (float)atof( token );
		}
		else if (!strcmp(token,"start"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			start = (float)atof( token );
		}
		else
			Error("Unknown command '%s' when loading Bounce!", token);
	}

}

// Applies the current transformation via glMultMatrix*()
void Bounce::Apply( void )
{
	float curValue = 0.5 * sin( start + speed*curTime - 1.570796 ) + 0.5;
	glTranslatef( curValue*dir.X(), curValue*dir.Y(), curValue*dir.Z() );
}

// Applies the current transformation to a vector
Vector Bounce::Apply( const Vector &vec )
{
	return vec;
}

// Applies the current transformation to a point
Point  Bounce::Apply( const Point &pt   )
{
	float curValue = 0.5 * sin( start + speed*curTime - 1.570796 ) + 0.5;
	return Point( pt.X()+curValue*dir.X(), pt.Y()+curValue*dir.Y(), pt.Z()+curValue*dir.Z() );
}

