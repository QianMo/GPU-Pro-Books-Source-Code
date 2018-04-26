/************************************************************************/
/* glLight.h                                                            */
/* ------------------                                                   */
/*                                                                      */
/* This file defines a class storing information about an OpenGL light, */
/*     as well as access routines.                                      */
/*                                                                      */
/* Chris Wyman (12/7/2007)                                              */
/************************************************************************/

#ifndef __GLLIGHT_H__
#define __GLLIGHT_H__

#include <stdio.h>
#include "Utils/GLee.h"
#include <GL/glut.h>
#include "DataTypes/Vector.h"
#include "DataTypes/Point.h"
#include "DataTypes/Matrix4x4.h"
#include "DataTypes/Color.h"
#include "Objects/ObjectMovement.h"

#pragma warning( disable: 4996 )

class Trackball;  
class Scene;
class GLSLProgram;

class GLLight
{
private:
	ObjectMovement *lightMove;

	// Point light basics
	Point pos, currentWorldPos;
	Point pointAt;
	Color amb, dif, spec;
	Vector spotDir;

	// GL data
	GLenum lightNum;
	float spotCutoff, spotExp;
	float atten[3];  // constant, linear, quadratic (resp. [0], [1], and [2])
	bool enabled;
	
	// If you decide to use this for shadow mapping (etc), you might need this.
	float _near, _far, fovy;

	// Movement of the light can be caused by a trackball (or matrix)
	Trackball *ball;
	Matrix4x4 *mat;

	// Misc data
	char *name;
public:
	// Setup defaults, associate this class with particular GL light.
	GLLight( GLenum lightNum=GL_LIGHT0 );
	GLLight( FILE *f, Scene *s );
	~GLLight();

	// Housekeeping details.  What GL light are we?
	inline int GetLightNum( void ) const { return lightNum; }

	// Every frame, the light position (and spot direction) may be modified by 
	//   different modelview matrices.  In order to account for this, thie method
	//   must be called each time a modelview matrix change would alter the light 
	//   position.  Usually that means this method should be called immediately
	//   after each call to gluLookAt().
	void SetLightUsingCurrentTransforms( void );

	// Functions to set various internal data.  Note:  These only call glLight*()
	//   to set OpenGL state if the calls are not dependant on current matrix state.
	void SetAmbient ( const Color &ambient )         { amb = ambient;   glLightfv ( lightNum, GL_AMBIENT, amb.GetDataPtr() );}
	void SetDiffuse ( const Color &diffuse )         { dif = diffuse;   glLightfv ( lightNum, GL_DIFFUSE, dif.GetDataPtr() );}
	void SetSpecular( const Color &specular )        { spec = specular; glLightfv ( lightNum, GL_SPECULAR, spec.GetDataPtr() );}
	void SetPosition( const Point &position )        { pos = position; }
	void SetPointAt ( const Point &newAt )           { pointAt = newAt; }
	//void SetSpotDirection( const Vector &spotDir )   { this->spotDir = spotDir; }
	void SetSpotCutoff( float degrees=180.0f );
	void SetSpotExponent( float exponent=0.0f );
	void SetAttenuation( float constant=1.0f, float linear=0.0f, float quadratic=0.0f );

	// Enable or disable the light
	inline void Enable( void )  { if (!enabled) glEnable( lightNum ); enabled=true;  }
	inline void Disable( void ) { if (enabled) glDisable( lightNum ); enabled=false; }
	inline bool IsEnabled( void ) const { return enabled; }

	// Perhaps one wants to use a trackball or other transformation to change the light position
	inline void AttachTrackball( Trackball *newBall ) { ball = newBall; }
	inline Trackball *GetTrackball( void ) { return ball; }
	inline void SetLightTransform( Matrix4x4 *xform ) { mat = xform; }

	// Get the current position of the light, modified by the trackball, if applicable
	const Point GetCurrentPos( void );
	const Point &GetOriginalPos( void )        { return pos; }

	// Get the position the light is pointing at
	inline const Point &GetLookAt( void )	   { return pointAt; }

	// These functions are needed to use this light for shadow maps (etc) where a
	//   perspective matrix is needed to render.  Otherwise, these parameters are
	//   ignored.
	inline void SetNearPlane( float newNear )    { _near = newNear; }
	inline void SetFarPlane( float newFar )      { _far = newFar; }
	inline void SetFOV(float s)                  { fovy = s; }
	inline float GetLightNearPlane( void ) const { return _near; }
	inline float GetLightFarPlane( void ) const  { return _far; }
	inline float GetLightFovy( void ) const      { return fovy; }

	// The light *might* have a name.  Access/change it:
	inline char *GetName( void )                 { return name; }
	inline void SetName( char *newName )         { if (name) free(name); name = strdup(newName); }

	void SetGLSLShaderCubeMapLookAtMatrices( GLSLProgram *shader, char *matrixArrayName );

	// Functions to see if geometry wants to update itself every frame.
	bool NeedPerFrameUpdates( void ) { return lightMove != 0; }
	void Update( float currentTime ) { if (lightMove) lightMove->Update( currentTime ); }
};



#endif