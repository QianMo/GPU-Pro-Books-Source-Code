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

#include <string.h>
#include "cScreenText.h"

//=======================================================================================
//
ScreenText::ScreenText( void )
{
/*
#   define  GLUT_STROKE_ROMAN               ((void *)0x0000)
#   define  GLUT_STROKE_MONO_ROMAN          ((void *)0x0001)
#   define  GLUT_BITMAP_9_BY_15             ((void *)0x0002)
#   define  GLUT_BITMAP_8_BY_13             ((void *)0x0003)
#   define  GLUT_BITMAP_TIMES_ROMAN_10      ((void *)0x0004)
#   define  GLUT_BITMAP_TIMES_ROMAN_24      ((void *)0x0005)
#   define  GLUT_BITMAP_HELVETICA_10        ((void *)0x0006)
#   define  GLUT_BITMAP_HELVETICA_12        ((void *)0x0007)
#   define  GLUT_BITMAP_HELVETICA_18        ((void *)0x0008)
*/
	projMan		= new ProjectionManager();
	font_style	= GLUT_BITMAP_9_BY_15;
	reset_l		= false;
	reset_t		= false;
}
//
//=======================================================================================
//
ScreenText::~ScreenText( void )
{
	FREE_INSTANCE( projMan );
}
//
//=======================================================================================
//
void ScreenText::setfont( char* name, int size )
{
    font_style = GLUT_BITMAP_HELVETICA_10;
    if( strcmp( name, "helvetica" ) == 0 )
	{
        if( size == 12 )
		{
            font_style = GLUT_BITMAP_HELVETICA_12;
		}
        else if( size == 18 )
		{
            font_style = GLUT_BITMAP_HELVETICA_18;
		}
    }
	else if( strcmp(name, "times roman" ) == 0 )
	{
        font_style = GLUT_BITMAP_TIMES_ROMAN_10;
        if( size == 24 )
		{
            font_style = GLUT_BITMAP_TIMES_ROMAN_24;
		}
    }
	else if( strcmp( name, "8x13" ) == 0 )
	{
        font_style = GLUT_BITMAP_8_BY_13;
    }
	else if( strcmp( name, "9x15" ) == 0 )
	{
        font_style = GLUT_BITMAP_9_BY_15;
    }
}
//
//=======================================================================================
//
void ScreenText::drawstr3D( GLuint x, GLuint y, GLuint z, char* format, ... )
{
    va_list args;
    char buffer[SCREEN_TEXT_BUFFER_SIZE], *s;

    va_start( args, format );
#if defined (_WIN32)
	vsprintf_s( buffer, SCREEN_TEXT_BUFFER_SIZE, format, args );
#elif defined (__unix)
	vsprintf( buffer, format, args );
#endif
    va_end( args );

    glRasterPos3i( x, y, z );
    for( s = buffer; *s; s++ )
	{
        glutBitmapCharacter( font_style, *s );
	}
}
//
//=======================================================================================
//
void ScreenText::begin( int w, int h )
{
	if( glIsEnabled( GL_LIGHTING ) )
	{
		reset_l = true;
		glDisable( GL_LIGHTING );
	}
	if( glIsEnabled( GL_TEXTURE_2D ) )
	{
		reset_t = true;
		glDisable( GL_TEXTURE_2D );
	}
	projMan->setTextProjection( w, h );
	glColor4f( 0.0f, 1.0f, 0.0f, 1.0f );
}
//
//=======================================================================================
//
void ScreenText::end( void )
{
	projMan->restoreProjection();
	if( reset_l )
	{
		glEnable( GL_LIGHTING );
	}
	reset_l = false;
	if( reset_t )
	{
		glEnable( GL_TEXTURE_2D );
	}
	reset_t = false;
	glColor4f( 1.0f, 1.0f, 1.0f, 1.0f );
}
//
//=======================================================================================
//
void ScreenText::drawstr2D( GLuint x, GLuint y, char* format, ... )
{
	glPushMatrix();
	{
		glLoadIdentity();
		va_list args;
		char buffer[SCREEN_TEXT_BUFFER_SIZE], *s;

		va_start( args, format );

#if defined (_WIN32)
		vsprintf_s( buffer, SCREEN_TEXT_BUFFER_SIZE, format, args );
#elif defined (__unix)
		vsprintf( buffer,format, args );
#endif
		va_end( args );

		glRasterPos2i( x, y );
		for( s = buffer; *s; s++ )
		{
			glutBitmapCharacter( font_style, *s );
		}
	}
	glPopMatrix();
}
//
//=======================================================================================
