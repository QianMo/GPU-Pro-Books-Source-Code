/************************************************************************/
/* areaLight.h                                                          */
/* ------------------                                                   */
/*                                                                      */
/* This file defines a class storing information about an area light,   */
/*     as well as access routines.                                      */
/*                                                                      */
/* Chris Wyman (05/12/2009)                                             */
/************************************************************************/

#ifndef __AREALIGHT_H__
#define __AREALIGHT_H__

#include <stdio.h>
#include "Utils/GLee.h"
#include <GL/glut.h>
#include "DataTypes/Vector.h"
#include "DataTypes/Point.h"
#include "DataTypes/Matrix4x4.h"
#include "DataTypes/Color.h"
#include "Objects/ObjectMovement.h"
#include "DataTypes/GLTexture.h"

#pragma warning( disable: 4996 )
 
class Scene;
class GLSLProgram;

class AreaLight
{
private:
	// Point light basics
	Point pos;
	Vector edge1, edge2;
	Vector surfNorm;
	GLTexture *lightTex;
	Color lightColor;
	
	// Misc data
	char *name;
public:
	// Setup defaults, associate this class with particular GL light.
	AreaLight( FILE *f, Scene *s );
	~AreaLight();

	// Functions to set various internal data.
	inline void SetAreaLightName( char *newName )			{ if (name) free(name); name = strdup(newName); }
	inline void SetAreaLightTexture( GLTexture *newTex )	{ lightTex = newTex; }
	inline void SetAreaLightColor( Color &newColor )		{ lightColor = newColor; }

	// Access light's internal data
	inline char *GetAreaLightName( void )					{ return name; }
	inline GLTexture *GetAreaLightTexture( void )			{ return lightTex; }
	inline Color &GetAreaLightColor( void )					{ return lightColor; }
	inline Point &GetAreaLightPosition( void )				{ return pos; }
	inline Vector &GetAreaLightEdge1( void )				{ return edge1; }
	inline Vector &GetAreaLightEdge2( void )				{ return edge2; }
	inline Vector &GetAreaLightNormal( void )				{ return surfNorm; }

	// Movement functions
	//  ... Stuff should go here ...

	// Functions to see if the light needs to update itself every frame.
	bool NeedPerFrameUpdates( void );
	void Update( float currentTime );
};



#endif