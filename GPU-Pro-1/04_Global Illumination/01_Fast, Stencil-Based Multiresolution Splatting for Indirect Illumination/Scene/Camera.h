/******************************************************************/
/* Camera.h                                                       */
/* -----------------------                                        */
/*                                                                */
/* The file defines a camera class that encapsulates infromation  */
/*    necessary for rendering with an OpenGL camera.              */
/*                                                                */
/* Chris Wyman (01/30/2008)                                       */
/******************************************************************/


#ifndef CAMERA_H
#define CAMERA_H 

#include "DataTypes/Vector.h"
#include "DataTypes/Point.h"
#include "DataTypes/MathDefs.h"
#include <stdio.h>
#include <math.h>
#include "Utils/Trackball.h"
#include "Objects/ObjectMovement.h"

class Scene;
class Matrix4x4;

class Camera {
private:
  Point eye, at;
  Vector up;
  float fovy, aspect, _near, _far;

  Trackball *ball;
  ObjectMovement *eyeMove;
public:

  /* sets up a camera */
  Camera() {}
  Camera( const Point &eye, const Point &at, const Vector &up, float fovy, float near, float far );
  Camera( FILE *f, Scene *s );

  /* accessor functions */
  inline Point GetEye( void ) const				{ return eye; }
  inline Point GetAt( void ) const				{ return at; }
  inline Vector GetUp( void ) const				{ return up; }
  inline Vector GetView( void ) const			{ return at-eye; }
  inline float GetFovy( void ) const			{ return fovy; }
  inline float GetTanFovx_2( void ) const		{ return tan( fovy * M_PI / 360.0f ) * aspect; }
  inline float GetTanFovy_2( void ) const		{ return tan( fovy * M_PI / 360.0f ); }
  inline float GetNear( void ) const			{ return _near; }
  inline float GetFar( void ) const				{ return _far; }
  inline float GetAspectRatio( void ) const		{ return aspect; }

  /* modification functions */
  inline void SetEye( const Point &newEye )		{ eye = newEye; }
  inline void SetAt( const Point &newAt )		{ at = newAt; }
  inline void SetUp( const Vector &newUp )		{ up = newUp; }
  inline void SetFovy( float newFovy )			{ fovy = newFovy; }
  inline void SetNear( float newNear )			{ _near = newNear; }
  inline void SetFar( float newFar )			{ _far = newFar; }
  inline void SetAspectRatio( float newAspect)	{ aspect = newAspect; }


	
  void MoveForward( float speed );
  void MoveBackwards( float speed );
  void MoveLeft( float speed );
  void MoveRight( float speed );
  void MoveUp( float speed );
  void MoveDown( float speed );

  // Set the OpenGL viewing parameters
  void LookAtMatrix( void );
  void InverseLookAtMatrix( void );  // Useful for shadow maps
  Matrix4x4 GetLookAtMatrix( void );
  Matrix4x4 GetInverseLookAtMatrix( void );

  // Get the camera's trackball (if any)
  inline Trackball *GetTrackball( void )        { return ball; }

  // Does this camera animate?
  bool NeedPerFrameUpdates( void ) { return eyeMove != 0; }
  void Update( float currentTime ) { if (eyeMove) eyeMove->Update( currentTime ); }

  // This may affect derived classes' internal state, so it needs to be virtual.
  // inline void SetCameraPosition( const Point &eye, const Point &at, const Vector &up ) = 0;

};


#endif

