/******************************************************************/
/* Camera.cpp                                                     */
/* -----------------------                                        */
/*                                                                */
/* The file defines a camera class that encapsulates infromation  */
/*    necessary for rendering with an OpenGL camera.              */
/*                                                                */
/* Chris Wyman (01/30/2008)                                       */
/******************************************************************/

#include "Camera.h"
#include "Scene.h"
#include "Utils/TextParsing.h"
#include "Utils/ImageIO/imageIO.h"
#include "DataTypes/Matrix4x4.h"


Camera::Camera( const Point &eye, const Point &at, const Vector &up, 
			   float fovy, float near, float far ) :
	eye(eye), at(at), up(up), fovy(fovy), _near(near), _far(far), eyeMove(0)
{
}


void Camera::LookAtMatrix( void )
{
	Vector rotVec = at-eye;
	rotVec = ( ball ? ball->ApplyTrackballMatrix( rotVec ) : rotVec );
	Point finalPos = ( eyeMove ? eyeMove->ApplyCurrentFrameMovementMatrix( at-rotVec ) : at-rotVec );
	gluLookAt( finalPos.X(), finalPos.Y(), finalPos.Z(), 
			   at.X(), at.Y(), at.Z(), 
			   up.X(), up.Y(), up.Z() );
}

Matrix4x4 Camera::GetLookAtMatrix( void )
{
	Vector rotVec = at-eye;
	rotVec = ( ball ? ball->ApplyTrackballMatrix( rotVec ) : rotVec );
	Point finalPos = ( eyeMove ? eyeMove->ApplyCurrentFrameMovementMatrix( at-rotVec ) : at-rotVec );
	return Matrix4x4::LookAt( finalPos, at, up );
}

Matrix4x4 Camera::GetInverseLookAtMatrix( void )
{
	Vector rotVec = at-eye;
	rotVec = ( ball ? ball->ApplyTrackballMatrix( rotVec ) : rotVec );
	Point finalPos = ( eyeMove ? eyeMove->ApplyCurrentFrameMovementMatrix( at-rotVec ) : at-rotVec );
	return Matrix4x4::LookAt( finalPos, at, up ).Invert();
}


void Camera::InverseLookAtMatrix( void )
{
	Vector rotVec = at-eye;
	rotVec = ( ball ? ball->ApplyTrackballMatrix( rotVec ) : rotVec );
	Point finalPos = ( eyeMove ? eyeMove->ApplyCurrentFrameMovementMatrix( at-rotVec ) : at-rotVec );
	Matrix4x4 m = Matrix4x4::LookAt( finalPos, at, up ).Invert();
	glMultMatrixf( m.GetDataPtr() );
}

void Camera::MoveForward( float speed )
{
	Vector forward( at-eye );
	forward.Normalize();
	eye = eye + speed*forward;
	at = at + speed*forward;
}

void Camera::MoveBackwards( float speed )
{
	Vector forward( at-eye );
	forward.Normalize();
	eye = eye - speed*forward;
	at = at - speed*forward;
}

void Camera::MoveUp( float speed )
{
	Vector forward( at-eye );
	forward.Normalize();
	eye = eye + speed*up;
	at = at + speed*up;
}

void Camera::MoveDown( float speed )
{
	eye = eye - speed*up;
	at = at - speed*up;
}


void Camera::MoveLeft( float speed )
{
	Vector forward( at-eye );
	forward.Normalize();
	Vector right( forward.Cross( up ) );
	right.Normalize();
	eye = eye - speed*right;
	at = at - speed*right;
}

void Camera::MoveRight( float speed )
{
	Vector forward( at-eye );
	forward.Normalize();
	Vector right( forward.Cross( up ) );
	right.Normalize();
	eye = eye + speed*right;
	at = at + speed*right;
}


Camera::Camera( FILE *f, Scene *s ) :
  fovy(90.f), eye( 0, 0, 1 ), at( 0, 0, 0 ),
  up( 0, 1, 0 ), _near(0.1), _far(20), ball(0), eyeMove(0)
{
	Matrix4x4 *mat=0;
	// Setup default values, in case the scene file is defective...
	s->SetWidth ( 512 );
	s->SetHeight( 512 );

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
		if (!strcmp(token,"eye")) 
			eye = Point( ptr ); 
		else if (!strcmp(token,"at")) 
			at = Point( ptr ); 
		else if (!strcmp(token,"up")) 
			up = Vector( ptr ); 
		else if (!strcmp(token,"w") || !strcmp(token,"width") )  
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			s->SetWidth( (int)atof( token ) );
		}
		else if (!strcmp(token,"h") || !strcmp(token,"height") )  
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			s->SetHeight( (int)atof( token ) );
		}
		else if (!strcmp(token,"res") || !strcmp(token, "resolution"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			s->SetWidth( (int)atof( token ) );
			ptr = StripLeadingTokenToBuffer( ptr, token );
			s->SetHeight( (int)atof( token ) );
		}
		else if (!strcmp(token, "matrix"))
		{
			if (!mat) 
				mat = new Matrix4x4( f, ptr );
			else
				(*mat) *= Matrix4x4( f, ptr );
		}
		else if (!strcmp(token,"trackball"))
			ball = new Trackball( s->GetWidth(), s->GetHeight() );
		else if (!strcmp(token,"fovy"))  
			ptr = StripLeadingNumber( ptr, &fovy );
		else if (!strcmp(token,"near"))  
			ptr = StripLeadingNumber( ptr, &_near );
		else if (!strcmp(token,"far"))  
			ptr = StripLeadingNumber( ptr, &_far );
		else if (!strcmp(token,"move") || !strcmp(token,"movement"))
			eyeMove = new ObjectMovement( f, s );
		else
			Error("Unknown command '%s' when loading Camera!", token);
	}

	if (mat)
		ball->SetTrackballMatrix( *mat );
}


