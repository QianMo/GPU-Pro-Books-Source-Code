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

#include "cAxes.h"

//=======================================================================================

Axes::Axes( float scale )
{
	axesList = glGenLists( 1 );
	glNewList( axesList, GL_COMPILE );
	{
		int x = 0, y = 0, z = 0;
		// X Axis:
		glColor3f( 1.0f, 0.0f, 0.0f );
		glBegin( GL_LINES );
		{
			glVertex3f( 0.0f, 0.0f, 0.0f );
			glVertex3f( scale, 0.0f, 0.0f );
		}
		glEnd();
		// Markers every 10 pixels:
		while( x <= scale )
		{
			glPushMatrix();
			{
				glTranslatef( (float)x, 0.0f, 0.0f );
				glutSolidSphere( 0.5f, 10, 10 );
				x += 10;
			}
			glPopMatrix();
		}
		// Y Axis:
		glColor3f( 0.0f, 1.0f, 0.0f );
		glBegin( GL_LINES );
		{
			glVertex3f( 0.0f, 0.0f, 0.0f );
			glVertex3f( 0.0f, scale, 0.0f );
		}
		glEnd();
		// Markers every 10 pixels:
		while( y <= scale )
		{
			glPushMatrix();
			{
				glTranslatef( 0.0f, (float)y, 0.0f );
				glutSolidSphere( 0.5f, 10, 10 );
				y += 10;
			}
			glPopMatrix();
		}
		// Z Axis:
		glColor3f( 0.0f, 0.0f, 1.0f );
		glBegin( GL_LINES );
		{
			glVertex3f( 0.0f, 0.0f, 0.0f );
			glVertex3f( 0.0f, 0.0f, scale );
		}
		glEnd();
		// Markers every 10 pixels:
		while( z <= scale )
		{
			glPushMatrix();
			{
				glTranslatef( 0.0f, 0.0f, (float)z );
				glutSolidSphere( 0.5f, 10, 10 );
				z += 10;
			}
			glPopMatrix();
		}
		// Negative axes:
		glColor3f( 0.5f, 0.5f, 0.5f );
		glBegin( GL_LINES );
		{
			glVertex3f(  0.0f,  0.0f, 0.0f );
			glVertex3f( -scale, 0.0f, 0.0f );
		}
		glEnd();
		glBegin( GL_LINES );
		{
			glVertex3f( 0.0f,  0.0f,  0.0f );
			glVertex3f( 0.0f, -scale, 0.0f );
		}
		glEnd();
		glBegin( GL_LINES );
		{
			glVertex3f( 0.0f, 0.0f,  0.0f  );
			glVertex3f( 0.0f, 0.0f, -scale );
		}
		glEnd();
	}
	glEndList();
}

//=======================================================================================

Axes::~Axes( void )
{
	glDeleteLists( axesList, 1 );
}

//=======================================================================================

void Axes::draw( void )
{
	glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT );
	{
		glDisable( GL_LIGHTING );
		glDisable( GL_TEXTURE_2D );
		glCallList( axesList );
	}
	glPopAttrib();
}

//=======================================================================================
