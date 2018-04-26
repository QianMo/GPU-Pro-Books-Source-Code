/******************************************************************/
/* Rotate.cpp                                                     */
/* -----------------------                                        */
/*                                                                */
/* Implements a simple rotation animation routine for obj updates */
/*                                                                */
/* Chris Wyman (6/30/2008)                                        */
/******************************************************************/

#include <stdio.h>
#include <math.h>
#include "Rotate.h"
#include "Utils/GLee.h"
#include <GL/glut.h>


// Constructor & Destructor
Rotate::Rotate( FILE *f ) : 
	start(0.0), speed(10.0), axis(0,1,0)
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
		if (!strcmp(token,"axis"))
			axis = Vector( ptr );
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
void Rotate::Apply( void )
{
	float curValue = start + ( 360.0 * curTime / speed ); 
	glRotatef( curValue, axis.X(), axis.Y(), axis.Z() );
}

// Applies the current transformation to a vector
Vector Rotate::Apply( const Vector &vec )
{
	float curValue = start + ( 360.0 * curTime / speed ); 
	Matrix4x4 mat( Matrix4x4::Rotate( curValue, axis ) );
	return mat * vec;
}

// Applies the current transformation to a point
Point  Rotate::Apply( const Point &pt   )
{
	float curValue = start + ( 360.0 * curTime / speed ); 
	Matrix4x4 mat( Matrix4x4::Rotate( curValue, axis ) );
	return mat * pt;
}

