/************************************************************************/
/* glLight.h                                                            */
/* ------------------                                                   */
/*                                                                      */
/* This file defines a class storing information about an OpenGL light, */
/*     as well as access routines.                                      */
/*                                                                      */
/* Chris Wyman (12/7/2007)                                              */
/************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "glLight.h"
#include "Scene/Scene.h"
#include "Utils/Trackball.h"
#include "Utils/TextParsing.h"
#include "DataTypes/Matrix4x4.h"
#include "DataTypes/Point.h"
#include "DataTypes/Vector.h"
#include "DataTypes/Color.h"
#include "Utils/glslProgram.h"

GLLight::GLLight( GLenum lightNum ) :
	lightNum(lightNum), enabled(false), ball(NULL),
		pos( 0, 1, 0 ), currentWorldPos( 0, 1, 0 ),
		amb( 0, 0, 0 ), dif( 0, 0, 0 ), spec( 0, 0, 0 ),
		spotDir( 0, 0, 0 ), spotExp(0), spotCutoff(180.0f),
		mat(0), _near(1), _far(50), fovy(90.0f), name(0), lightMove(0)
{
	// Default OpenGL light attenuation
	atten[0] = 1.0f; atten[1] = atten[2] = 0.0f;

	// GL_LIGHT0 has special defaults...
	if (lightNum == GL_LIGHT0)
	{
		dif = Color::White();
		spec = Color::White();
	}
}

GLLight::GLLight( FILE *f, Scene *s ) :
	lightNum(0), enabled(false), ball(NULL),
		pos( 0, 1, 0 ), currentWorldPos( 0, 1, 0 ),
		amb( 0, 0, 0 ), dif( 0, 0, 0 ), spec( 0, 0, 0 ),
		spotDir( 0, 0, 0 ), spotExp(0), spotCutoff(180.0f),
		mat(0), _near(1), _far(50), fovy(90.0f), name(0),
		lightMove(0)
{ 
	lightNum = s->GetNumLights() + GL_LIGHT0;

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
		if ( !strcmp(token,"rgb") ||      // This means there's only one light "color"
			 !strcmp(token,"color") ||    //   so assign amb, dif, and spec colors to 
			 !strcmp(token,"spectral"))   //   this value.
		{
			 dif = LoadColor::Load( token, f, ptr );
			 amb = dif; spec = dif;
		}
		else if (!strcmp(token,"amb") || !strcmp(token,"ambient") )  
			amb = LoadColor::Load( token, f, ptr );
		else if (!strcmp(token,"dif") || !strcmp(token,"diffuse") )  
			dif = LoadColor::Load( token, f, ptr );
		else if (!strcmp(token,"spec") || !strcmp(token,"specular") )  
			spec = LoadColor::Load( token, f, ptr );
		else if (!strcmp(token,"pos") || !strcmp(token,"position") )  
			pos = Point( ptr );
		else if (!strcmp(token,"at") || !strcmp(token,"pointat") )  
			pointAt = Point( ptr );
		else if (!strcmp(token,"near"))  
			ptr = StripLeadingNumber( ptr, &_near );
		else if (!strcmp(token,"far"))  
			ptr = StripLeadingNumber( ptr, &_far );
		else if (!strcmp(token, "matrix") || !strcmp(token, "xform"))
		{
			if (!mat) 
				mat = new Matrix4x4( f, ptr );
			else
				(*mat) *= Matrix4x4( f, ptr );
		}
		else if (!strcmp(token, "trackball"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			int id = atoi(token);
			ball = new Trackball( s->GetWidth(), s->GetHeight() );
			s->SetupLightTrackball( id, ball );
		}
		else if (!strcmp(token,"move") || !strcmp(token,"movement"))
		{
			lightMove = new ObjectMovement( f, s );
		}
		else if (!strcmp(token,"name"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			this->SetName( token );
		}
		else
			printf("Error: Unknown command '%s' when loading Light\n", token);
	}


	
	currentWorldPos = pos;
}



GLLight::~GLLight()
{
	if( mat) delete mat;
}

void GLLight::SetAttenuation( float constant, float linear, float quadratic )
{
	atten[0] = constant; 
	atten[1] = linear; 
	atten[2] = quadratic;
	glLightf( lightNum, GL_CONSTANT_ATTENUATION, atten[0] );
	glLightf( lightNum, GL_LINEAR_ATTENUATION, atten[1] );
	glLightf( lightNum, GL_QUADRATIC_ATTENUATION, atten[2] );
}

void GLLight::SetSpotCutoff( float degrees )
{
	spotCutoff = degrees;
	glLightf( lightNum, GL_SPOT_CUTOFF, spotCutoff );
}

void GLLight::SetSpotExponent( float exponent ) 
{ 
	spotExp = exponent; 
	glLightf( lightNum, GL_SPOT_EXPONENT, spotExp );
}

void GLLight::SetLightUsingCurrentTransforms( void )
{
	if (mat || ball || lightMove)
	{
		glPushMatrix();
		glTranslatef( pointAt.X(), pointAt.Y(), pointAt.Z() );
		if (lightMove) lightMove->ApplyCurrentFrameMovementMatrix();
		if (mat)       glMultMatrixf( mat->GetDataPtr() );
		if (ball)      ball->MultiplyTrackballMatrix();
		glTranslatef( -pointAt.X(), -pointAt.Y(), -pointAt.Z() );
	}
	glLightfv( lightNum, GL_POSITION, pos.GetDataPtr() );
	glLightfv( lightNum, GL_SPOT_DIRECTION, pos.GetDataPtr() );
	if (mat || ball || lightMove)
		glPopMatrix();
}



const Point GLLight::GetCurrentPos( void )        
{
	Point pt = (ball ? ball->ApplyTrackballMatrix( pos-pointAt ) : pos-pointAt );
	pt = (mat ? (*mat)*pt : pt);
	pt = (lightMove ? lightMove->ApplyCurrentFrameMovementMatrix( pt ) : pt );
	return pt+pointAt;
}


void GLLight::SetGLSLShaderCubeMapLookAtMatrices( GLSLProgram *shader, char *matrixArrayName )
{
	bool enabled = shader->IsEnabled();
	float matrix[16];
	char buf[128];
	Point curpos = GetCurrentPos();
	if (!enabled) shader->EnableShader();
	glPushMatrix();
	sprintf(buf, "%s[0]", matrixArrayName );
	glLoadIdentity();
	gluLookAt( curpos.X(), curpos.Y(), curpos.Z(), curpos.X()+1, curpos.Y(), curpos.Z(), 0, 1, 0 );
	glGetFloatv( GL_MODELVIEW_MATRIX, matrix );
	shader->Set4x4MatrixParameterv( buf, matrix );
	sprintf(buf, "%s[1]", matrixArrayName );
	glLoadIdentity();
	gluLookAt( curpos.X(), curpos.Y(), curpos.Z(), curpos.X()-1, curpos.Y(), curpos.Z(), 0, 1, 0 );
	glGetFloatv( GL_MODELVIEW_MATRIX, matrix );
	shader->Set4x4MatrixParameterv( buf, matrix );
	sprintf(buf, "%s[2]", matrixArrayName );
	glLoadIdentity();
	gluLookAt( curpos.X(), curpos.Y(), curpos.Z(), curpos.X(), curpos.Y()+1, curpos.Z(), 0, 0, 1 );
	glGetFloatv( GL_MODELVIEW_MATRIX, matrix );
	shader->Set4x4MatrixParameterv( buf, matrix );
	sprintf(buf, "%s[3]", matrixArrayName );
	glLoadIdentity();
	gluLookAt( curpos.X(), curpos.Y(), curpos.Z(), curpos.X(), curpos.Y()-1, curpos.Z(), 0, 0, 1 );
	glGetFloatv( GL_MODELVIEW_MATRIX, matrix );
	shader->Set4x4MatrixParameterv( buf, matrix );
	sprintf(buf, "%s[4]", matrixArrayName );
	glLoadIdentity();
	gluLookAt( curpos.X(), curpos.Y(), curpos.Z(), curpos.X(), curpos.Y(), curpos.Z()+1, 1, 0, 0 );
	glGetFloatv( GL_MODELVIEW_MATRIX, matrix );
	shader->Set4x4MatrixParameterv( buf, matrix );
	sprintf(buf, "%s[5]", matrixArrayName );
	glLoadIdentity();
	gluLookAt( curpos.X(), curpos.Y(), curpos.Z(), curpos.X(), curpos.Y(), curpos.Z()-1, 1, 0, 0 );
	glGetFloatv( GL_MODELVIEW_MATRIX, matrix );
	shader->Set4x4MatrixParameterv( buf, matrix );
	glPopMatrix();
	if (!enabled) shader->DisableShader();
}

