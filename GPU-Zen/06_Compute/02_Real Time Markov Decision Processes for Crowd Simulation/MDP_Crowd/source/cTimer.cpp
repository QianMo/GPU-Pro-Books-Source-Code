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


#include "cTimer.h"

//=======================================================================================
//
Timer* Timer::singleton = 0;
//
//=======================================================================================
//
Timer::Timer( void )
{
	init();
}
//
//=======================================================================================
//
Timer::~Timer( void )
{

}
//
//=======================================================================================
//
Timer* Timer::getInstance( void )
{
	if( singleton == 0 )
	{
		singleton = new Timer;
	}

	return ((Timer *)singleton);
}
//
//=======================================================================================
//
void Timer::freeInstance( void )
{
	if( singleton != 0 )
	{
		FREE_INSTANCE( singleton );
	}
}
//
//=======================================================================================
//
void Timer::init( void )
{
	currentTime = getTimeMSec();
	lastTime	= 0;
}
//
//=======================================================================================
//
void Timer::update( void )
{
	lastTime	= currentTime;
	currentTime = getTimeMSec();
}
//
//=======================================================================================
//
unsigned long Timer::getTimeMSec( void )
{
	return glutGet( GLUT_ELAPSED_TIME );
}
//
//=======================================================================================
//
unsigned long Timer::getTime( void )
{
	return currentTime;
}
//
//=======================================================================================
//
float Timer::getAnimTime( void )
{
	return ((float)glutGet( GLUT_ELAPSED_TIME ) * 0.001f);
}
//
//=======================================================================================
//
float Timer::getFps( void )
{
	return (1000.0f / (float)(currentTime - lastTime));
}
//
//=======================================================================================
//
#ifdef _WIN32
void Timer::start( DWORD& time1 )
{
	time1 = GetTickCount();
}
//
//=======================================================================================
//
float Timer::stop( DWORD& time1 )
{
	DWORD init_elapsedTicks = GetTickCount() - time1;
	return (float)init_elapsedTicks;
}
#elif defined __unix
//
//=======================================================================================
//
void Timer::start( timeval& time1 )
{
	gettimeofday( &time1, NULL );
}
//
//=======================================================================================
//
float Timer::stop( timeval&	time1 )
{
	timeval stop, result;
	gettimeofday( &stop, NULL );
	timersub( &stop, &time1, &result );
	return (float)result.tv_sec*1000.0f+(float)result.tv_usec/1000.0f;
}
#endif
//
//=======================================================================================
