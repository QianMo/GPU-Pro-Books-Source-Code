#include "App.h"

#ifdef _WIN32
#include <windows.h>
#include <gl/gl.h>
#else
#include <opengl/gl.h>
#endif

//---------------------------------------------------------------------------------------------------------------------------------

App* App::m_app = 0;

//---------------------------------------------------------------------------------------------------------------------------------

App::App( void )
{
	m_app = this;
}

//---------------------------------------------------------------------------------------------------------------------------------

App::~App( void )
{

}

//---------------------------------------------------------------------------------------------------------------------------------

void App::HandleInput( unsigned char key )
{
	Vector4 velocity = Vector4::ZERO;
	float rot_x = 0;
	float rot_y = 0;

	switch( key )
	{
		case 'w':
			velocity.SetZ( -1.0f );
		break;
		
		case 's':
			velocity.SetZ( +1.0f );
		break;
		
		case 'a':
			velocity.SetX( -1.0f );
		break;
		
		case 'd':
			velocity.SetX( +1.0f );
		break;
		
		case 'j':
			rot_y = -0.02f;
		break;
		
		case 'l':
			rot_y = +0.02f;
		break;
		
		case 'i':
			rot_x = +0.02f;
		break;
		
		case 'k':
			rot_x = -0.02f;
		break;
		
		case 'y':
			m_crowd.AdjustDensity( +1 );
		break;
		
		case 'h':
			m_crowd.AdjustDensity( -1 );
		break;
	}

	m_camera.Move( velocity );
	m_camera.RotateX( rot_x );
	m_camera.RotateY( rot_y );
}

//---------------------------------------------------------------------------------------------------------------------------------

void App::Initialize( void )
{
	//Initialize the shader system
	Shader::Initialize();
	
	m_camera.SetPosition( Vector4( 0.0f, 3.0f, 0.0f, 1.0f ) );
	m_camera.RotateY( 3.1416f );
	m_camera.Create( 45.0f, 640.0f / 480.0f, 0.1f, 10000.0f );
	m_stadium.Create( 100, 10, 50 );
	m_crowd.Create( 99, 10, 49, 10 );

	glClearColor( 97.0f / 255.0f, 148.0f / 255.0f, 226.0f / 255.0f, 0.0f );	
}

//---------------------------------------------------------------------------------------------------------------------------------

void App::Render( void )
{
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );
	glDepthFunc( GL_LESS );
	glPixelStorei( GL_UNPACK_ALIGNMENT, 4 );
	glFrontFace( GL_CCW );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glLineWidth( 1 );
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
	glDepthMask( GL_TRUE );
	glAlphaFunc( GL_NOTEQUAL, 0.0f );
	glEnable( GL_DEPTH_TEST );
	glEnable( GL_ALPHA_TEST );
	glEnable( GL_CULL_FACE );
	glDisable( GL_SCISSOR_TEST );	
	glDisable( GL_STENCIL_TEST );
	glDisable( GL_BLEND );
	glPointSize( 3.0f );

	m_stadium.Render( m_camera );
	m_crowd.Render( m_camera );

	glFlush();
	glFinish();
}

//---------------------------------------------------------------------------------------------------------------------------------

void App::Update( float dt )
{
	m_camera.Update();
	m_crowd.Update( dt );
}

//---------------------------------------------------------------------------------------------------------------------------------

void App::Shutdown( void )
{
	m_stadium.Shutdown();
	m_crowd.Shutdown();
}

//---------------------------------------------------------------------------------------------------------------------------------
