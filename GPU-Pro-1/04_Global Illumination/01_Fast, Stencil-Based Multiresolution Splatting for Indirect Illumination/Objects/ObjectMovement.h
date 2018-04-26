/******************************************************************/
/* ObjectMovement.h                                               */
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

#ifndef __OBJECTMOVEMENT_H__
#define __OBJECTMOVEMENT_H__

#include <stdio.h>
#include "Deformations/ObjectTransform.h"

class Scene;
class Vector;
class Point;

class ObjectMovement {
private: 
	ObjectTransform *translate;
	ObjectTransform *rotate;
	ObjectTransform *scale;
public:
	// Constructors & destructors
	ObjectMovement() : translate(0), rotate(0), scale(0) {}
	ObjectMovement( FILE *f, Scene *s );
	~ObjectMovement() {}

	// Updates internal variables to represent new object transform
	void Update( float currentTime );

	// We're about ready to draw the moved object.  Apply the currently calculated
	//    matrix transformations for the current timestep.
	void   ApplyCurrentFrameMovementMatrix( void );              // Multiply onto the stack
	Vector ApplyCurrentFrameMovementMatrix( const Vector &vec ); // Multiply the given vector
	Point  ApplyCurrentFrameMovementMatrix( const Point &pt );   // Multiply the given point

	// Sometimes, not all of the transformations are desired.  Perhaps only the 
	//    translation component or only the scale component or only the rotation 
	//    is desired (or the ordering needs to be controlled more finely).  These
	//    can be individually accessed using the following methods.
	void ApplyCurrentFrameTranslationMatrix( void );
	void ApplyCurrentFrameScaleMatrix( void );
	void ApplyCurrentFrameRotationMatrix( void );

	Vector ApplyCurrentFrameTranslationMatrix( const Vector &vec );
	Vector ApplyCurrentFrameScaleMatrix( const Vector &vec );
	Vector ApplyCurrentFrameRotationMatrix( const Vector &vec );

	Point ApplyCurrentFrameTranslationMatrix( const Point &pt );
	Point ApplyCurrentFrameScaleMatrix( const Point &pt );
	Point ApplyCurrentFrameRotationMatrix( const Point &pt );
};


#endif