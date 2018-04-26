/************************************************************************/
/* drawTextToGLWindow.cpp                                               */
/* ------------------                                                   */
/*                                                                      */
/* This file contains a number of utility functions that                */
/*    use GLUT to draw text into the OpenGL window.                     */
/*                                                                      */
/* Chris Wyman (12/7/2007)                                              */
/************************************************************************/

#include "drawTextToGLWindow.h"

#pragma warning( disable: 4996 )


void PrintString(char *str, void *font)        
{                                     
  int len, i;                        
                                   
  len = (int) strlen(str);
  for(i=0; i<len; i++)
        glutBitmapCharacter(font, str[i]);
}


void DisplayString( int rasterPosX, int rasterPosY, char *str, int screenWidth, int screenHeight )
{
	int width=screenWidth, height=screenHeight;

	// If user didn't kindly give us the screen resolution, find it.
	if (width <= 0 || height <= 0)
	{
		width = glutGet( GLUT_WINDOW_WIDTH );
		height = glutGet( GLUT_WINDOW_HEIGHT );
	}

	glPushAttrib( GL_ALL_ATTRIB_BITS );
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D( 0.0, width, 0.0, height );
	glMatrixMode( GL_MODELVIEW );

	glPushMatrix();
	glLoadIdentity();
	glRasterPos2i(rasterPosX, rasterPosY);
	PrintString( str );
	glPopMatrix();

	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopAttrib();
}


void DisplayTimer( float fps, int screenWidth, int screenHeight )
{
	char buf[1024];
	sprintf( buf, "%.2f fps", fps );
	glColor3f(0,0,0);
	DisplayString( 2, 9, buf, screenWidth, screenHeight );
	glColor3f(1,1,1);
	DisplayString( 3, 10, buf, screenWidth, screenHeight );
}



