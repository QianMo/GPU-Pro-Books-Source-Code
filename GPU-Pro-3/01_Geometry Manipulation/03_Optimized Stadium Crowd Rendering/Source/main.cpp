/**
 *	@file
 *	@brief		Non-optimized matrix class
 *	@author		Alan Chambers
 *	@date		2011
**/

#include "App.h"
#include "Shader.h"

#ifdef _WIN32
#include <gl/glut.h>
#include <gl/glew.h>
#else
#include <glut/glut.h>
#endif

//---------------------------------------------------------------------------------------------------------------------------------

const float cScreenWidth = 640.0f;
const float cScreenHeight = 480.0f;

//---------------------------------------------------------------------------------------------------------------------------------

void display( void )
{
	float dt = 1.0f / 60.0f; //Fixed game time loop
	
	App::Get()->Update( dt );
	App::Get()->Render();
	
	glutSwapBuffers();
}

//---------------------------------------------------------------------------------------------------------------------------------

void reshape( int width, int height )
{
    glViewport( 0, 0, width, height );
}

//---------------------------------------------------------------------------------------------------------------------------------

void idle( void )
{
	glutPostRedisplay();
}

//---------------------------------------------------------------------------------------------------------------------------------

void keyboard( unsigned char key, int, int )
{
	App::Get()->HandleInput( key );
}

//---------------------------------------------------------------------------------------------------------------------------------

int main( int argc, char** argv )
{
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );
    glutInitWindowSize( cScreenWidth, cScreenHeight );
    glutCreateWindow( "Optimized Stadium Crowd Rendering" );
    glutDisplayFunc( display );
    glutReshapeFunc( reshape );
    glutIdleFunc( idle );
	glutIgnoreKeyRepeat( 0 );
	glutSetKeyRepeat( 1 );
	glutKeyboardFunc( keyboard );

#ifdef _WIN32
	GLenum ret = glewInit();
	if( ret != GLEW_OK )
		return -1;

	if( !glewIsExtensionSupported( "GL_ARB_draw_instanced" ) )
	{
		MessageBox( NULL, "Graphics card does not support instancing. The application will now exit.", "Unsupported Extension", MB_OK | MB_ICONERROR );
		return -1;
	}

	if( !Shader::IsProfileSupported( "gp4vp" ) )
	{
		MessageBox( NULL, "Graphics card does not support gp4 profile. The application will now exit.", "Unsupported Shader Profile", MB_OK | MB_ICONERROR );
		return -1;
	}
#else
	if( !Shader::IsProfileSupported( "gp4vp" ) )
	{
		printf( "Graphics card does not support gp4 profile. The application will now exit." );
		return -1;
	}
#endif

	App app;
	
	app.Initialize();
	
    glutMainLoop();
	
	app.Shutdown();
	
    return 0;
}

//---------------------------------------------------------------------------------------------------------------------------------
