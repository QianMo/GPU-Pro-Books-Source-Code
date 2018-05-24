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
// Source: http://www.opengl.org/sdk/docs/man/xhtml/glGetError.xml

#include "cGlErrorManager.h"

//=======================================================================================
//
GlErrorManager::GlErrorManager( LogManager* log_manager_ )
{
    log_manager = log_manager_;
	err			= 0;
	first		= 0;
}
//
//=======================================================================================
//
GlErrorManager::~GlErrorManager( void )
{

}
//
//=======================================================================================
//
GLenum GlErrorManager::getError( void )
{
	err		= glGetError();
	first	= err;
	do
	{
		if( err != 0 )
		{
			getString();
			if( log_manager )
				log_manager->log( LogManager::LERROR, "(%i)%s", err, str_err.c_str() );
            else
				printf( "(%i)%s\n", err, str_err.c_str() );
		}
		err = glGetError();
	}
	while( err != GL_NO_ERROR );
	return first;
}
//
//=======================================================================================
//
GLenum GlErrorManager::getError( const char* note )
{
	err		= glGetError();
	first	= err;
	do
	{
		if( err != 0 )
		{
			getString();
			if( log_manager )
				log_manager->log( LogManager::LERROR, "%s::(%i)%s", note, err, str_err.c_str() );
            else
				printf( "%s\n(%i)%s\n", note, err, str_err.c_str() );
		}
		err = glGetError();
	}
	while( err != GL_NO_ERROR );
	return first;
}
//
//=======================================================================================
//
void GlErrorManager::getString( void )
{
	switch( err )
	{
		case GL_NO_ERROR:
			str_err = string(	"\t\t\tGL_NO_ERROR:\n\t\t\tNo error has been recorded.\n\t\t\tThe "
								"value of this symbolic constant is guaranteed to be 0."			);
			break;
		case GL_INVALID_ENUM:
			str_err = string(	"\t\t\tGL_INVALID_ENUM:\n\t\t\tAn unacceptable value is specified "
								"for an enumerated argument.\n\t\t\tThe offending command is ignored"
								"\n\t\t\tand has no other side effect than to set the error flag."	);
			break;
		case GL_INVALID_VALUE:
			str_err = string(	"\t\t\tGL_INVALID_VALUE:\n\t\t\tA numeric argument is out of range."
								"\n\t\t\tThe offending command is ignored\n\t\t\tand has no other "
								"side effect than to set the error flag."							);
			break;
		case GL_INVALID_OPERATION:
			str_err = string(	"\t\t\tGL_INVALID_OPERATION:\n\t\t\tThe specified operation is not "
								"allowed in the current state.\n\t\t\tThe offending command is "
								"ignored\n\t\t\tand has no other side effect than to set the "
								"error flag."														);
			break;
		case GL_STACK_OVERFLOW:
			str_err = string(	"\t\t\tGL_STACK_OVERFLOW:\n\t\t\tThis command would cause a stack "
								"overflow.\n\t\t\tThe offending command is ignored\n\t\t\tand has "
								"no other side effect than to set the error flag."					);
			break;
		case GL_STACK_UNDERFLOW:
			str_err = string(	"\t\t\tGL_STACK_UNDERFLOW:\n\t\t\tThis command would cause a stack "
								"underflow.\n\t\t\tThe offending command is ignored\n\t\t\tand has "
								"no other side effect than to set the error flag."					);
			break;
		case GL_OUT_OF_MEMORY:
			str_err = string(	"\t\t\tGL_OUT_OF_MEMORY:\n\t\t\tThere is not enough memory left to "
								"execute the command.\n\t\t\tThe state of the GL is undefined,"
								"\n\t\t\texcept for the state of the error flags,\n\t\t\tafter this "
								"error is recorded."												);
			break;
		case GL_TABLE_TOO_LARGE:
			str_err = string(	"\t\t\tGL_TABLE_TOO_LARGE:\n\t\t\tThe specified table exceeds the "
								"implementation's maximum supported table\n\t\t\tsize.  The offending"
								" command is ignored and has no other side effect\n\t\t\tthan to set "
								"the error flag."													);
			break;
		default:
			str_err = string(	"\t\t\tUNKNOWN_ERROR!"												);
			break;
	}
}
//
//=======================================================================================
