/******************************************************************/
/* ObjectMovement.cpp                                             */
/* -----------------------                                        */
/*                                                                */
/* The file defines an animation class for objects.  Ideally,     */
/*    this class will allow arbitrary translation, rotation, and  */
/*    scaling (as well as combinations) to scene geometry as time */
/*    changes.  However, not all functionality is yet defined.    */
/*                                                                */
/* As with the rest of this code, the constructor reads movement  */
/*    specified by the scene file.  Updates occur by calling      */
/*    scene->PerFrameUpdate( currentTime ), which in turn calls   */
/*    each individual object's update routine, which calls the    */
/*    Update() method for this class.                             */
/*                                                                */
/* Chris Wyman (6/30/2008)                                        */
/******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "Utils/GLee.h"
#include <GL/glut.h>
#include "DataTypes/Vector.h"
#include "DataTypes/Point.h"
#include "DataTypes/Matrix4x4.h"
#include "ObjectMovement.h"
#include "Scene/Scene.h"
#include "Deformations/Bounce.h"
#include "Deformations/Rotate.h"

// Prototype for Error() used inside the constructor...
//    Prototype originally defined in ImageIO/ImageIO.h
void Error( char *formatStr, char *insertThisStr );



ObjectMovement::ObjectMovement( FILE *f, Scene *s ) :
	translate(0), rotate(0), scale(0)
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
		if (!strcmp(token,"bounce"))
			translate = new Bounce( f );
		else if (!strcmp(token,"rotate"))
			rotate = new Rotate( f );
		else
			Error("Unknown command '%s' when loading ObjectMovement()!", token);
	}

}


void ObjectMovement::Update( float currentTime )
{
	if (translate) translate->Update( currentTime );
	if (rotate)    rotate->Update( currentTime );
	if (scale)     scale->Update( currentTime );
}


void ObjectMovement::ApplyCurrentFrameMovementMatrix( void )
{
	ApplyCurrentFrameTranslationMatrix();
	ApplyCurrentFrameRotationMatrix();
	ApplyCurrentFrameScaleMatrix();
}

Vector ObjectMovement::ApplyCurrentFrameMovementMatrix( const Vector &vec )
{
	return ApplyCurrentFrameTranslationMatrix(
				ApplyCurrentFrameRotationMatrix(
					ApplyCurrentFrameScaleMatrix( vec ) ) );
}

Point  ObjectMovement::ApplyCurrentFrameMovementMatrix( const Point &pt )
{
	return ApplyCurrentFrameTranslationMatrix(
				ApplyCurrentFrameRotationMatrix(
					ApplyCurrentFrameScaleMatrix( pt ) ) );
}

void ObjectMovement::ApplyCurrentFrameTranslationMatrix( void )
{
	if (translate) translate->Apply();
}

void ObjectMovement::ApplyCurrentFrameScaleMatrix( void )
{
	if (rotate) rotate->Apply();
}

void ObjectMovement::ApplyCurrentFrameRotationMatrix( void )
{
	if (scale) scale->Apply();
}

Vector ObjectMovement::ApplyCurrentFrameTranslationMatrix( const Vector &vec )
{
	return (translate ? translate->Apply(vec) : vec);
}

Vector ObjectMovement::ApplyCurrentFrameScaleMatrix( const Vector &vec )
{
	return (scale ? scale->Apply(vec) : vec);
}

Vector ObjectMovement::ApplyCurrentFrameRotationMatrix( const Vector &vec )
{
	return (rotate ? rotate->Apply(vec) : vec);
}

Point ObjectMovement::ApplyCurrentFrameTranslationMatrix( const Point &pt )
{
	return (translate ? translate->Apply(pt) : pt);
}

Point ObjectMovement::ApplyCurrentFrameScaleMatrix( const Point &pt )
{
	return (scale ? scale->Apply(pt) : pt);
}

Point ObjectMovement::ApplyCurrentFrameRotationMatrix( const Point &pt )
{
	return (rotate ? rotate->Apply(pt) : pt);
}




