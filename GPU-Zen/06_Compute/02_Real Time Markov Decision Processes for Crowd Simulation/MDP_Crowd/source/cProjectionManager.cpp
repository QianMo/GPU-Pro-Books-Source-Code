/*

Copyright 2013,2014 Sergio Ruiz, Benjamin Hernandez

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

In case you, or any of your employees or students, publish any article or
other material resulting from the use of this  software, that publication
must cite the following references:

Sergio Ruiz, Benjamin Hernandez, Adriana Alvarado, and Isaac Rudomin. 2013.
Reducing Memory Requirements for Diverse Animated Crowds. In Proceedings of
Motion on Games (MIG '13). ACM, New York, NY, USA, , Article 55 , 10 pages.
DOI: http://dx.doi.org/10.1145/2522628.2522901

Sergio Ruiz and Benjamin Hernandez. 2015. A Parallel Solver for Markov Decision Process
in Crowd Simulations. Fourteenth Mexican International Conference on Artificial
Intelligence (MICAI), Cuernavaca, 2015, pp. 107-116.
DOI: 10.1109/MICAI.2015.23

*/

#include "cProjectionManager.h"
//
//=======================================================================================
//
ProjectionManager::ProjectionManager( void )
{
	backup_viewport = false;
	type			= ORTHOGRAPHIC;
}
//
//=======================================================================================
//
ProjectionManager::~ProjectionManager( void )
{

}
//
//=======================================================================================
//
void ProjectionManager::setOrthoProjection( unsigned int	w,
											unsigned int	h,
											bool			backup_vp
										  )
{
	type			= ORTHOGRAPHIC;
	backup_viewport	= backup_vp;
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
		glLoadIdentity();
		if( backup_viewport )
		{
			glGetIntegerv( GL_VIEWPORT, &viewport[0] );
			glViewport( 0, 0, (GLsizei)w, (GLsizei)h );
		}
		glOrtho( 0.0, (double)w, 0.0, (double)h, -1.0, 1.0 );
		glMatrixMode( GL_MODELVIEW );
		glPushMatrix();
			glLoadIdentity();
}
//
//=======================================================================================
//
void ProjectionManager::setOrthoProjection( unsigned int	x,
											unsigned int	y,
											unsigned int    w,
											unsigned int    h,
											unsigned int	l,
											unsigned int	r,
											unsigned int    b,
											unsigned int    t,
											bool			backup_vp
										  )
{
	type			= ORTHOGRAPHIC;
	backup_viewport	= backup_vp;
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
		glLoadIdentity();
		if( backup_viewport )
		{
			glGetIntegerv( GL_VIEWPORT, &viewport[0] );
			glViewport( x, y, (GLsizei)w, (GLsizei)h );
		}
		glOrtho( (double)l, (double)r, (double)b, (double)t, -1.0, 1.0 );
		glMatrixMode( GL_MODELVIEW );
		glPushMatrix();
			glLoadIdentity();
}
//
//=======================================================================================
//
void ProjectionManager::setTextProjection( unsigned int w, unsigned int h )
{
	type = TEXT;
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
		glLoadIdentity();
		gluOrtho2D( 0, w, h, 0 );
		glMatrixMode( GL_MODELVIEW );
}
//
//=======================================================================================
//
void ProjectionManager::restoreProjection( void )
{
	if( type == ORTHOGRAPHIC )
	{
			glPopMatrix();
			glMatrixMode( GL_PROJECTION );
			if( backup_viewport )
			{
				glViewport( viewport[0],
							viewport[1],
							(GLsizei)viewport[2],
							(GLsizei)viewport[3] );
			}
		glPopMatrix();
		glMatrixMode( GL_MODELVIEW );
	}
	else if( type == TEXT )
	{
			glMatrixMode( GL_PROJECTION );
		glPopMatrix();
		glMatrixMode( GL_MODELVIEW );
	}
}
//
//=======================================================================================
//
GLint* ProjectionManager::getViewport( void )
{
	glGetIntegerv( GL_VIEWPORT, &viewport[0] );
	return viewport;
}
//
//=======================================================================================
//
float ProjectionManager::getAspectRatio( void )
{
	glGetIntegerv( GL_VIEWPORT, &viewport[0] );
	aspect_ratio = (float)viewport[2] / (float)viewport[3];
	return aspect_ratio;
}
//
//=======================================================================================
//
void ProjectionManager::displayText( int x, int y, char* txt )
{
	GLint viewportCoords[ 4 ];
	glColor3f( 0.0, 0.0, 1.0 );
	glGetIntegerv( GL_VIEWPORT, viewportCoords );

	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D( 0.0, viewportCoords[2], 0.0, viewportCoords[3] );
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	glRasterPos2i( x, viewportCoords[3]-y );

	while (*txt) {glutBitmapCharacter( GLUT_BITMAP_9_BY_15, *txt ); txt++;}
	glPopMatrix();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
}
//
//=======================================================================================
