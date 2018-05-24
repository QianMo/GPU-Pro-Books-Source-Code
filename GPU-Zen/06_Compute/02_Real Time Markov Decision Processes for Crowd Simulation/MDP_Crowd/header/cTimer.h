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

#pragma once

#ifdef __unix
	#include <sys/time.h>
#endif
#include "cMacros.h"

//=======================================================================================

#ifndef		__TIMER
#define		__TIMER

class Timer
{
protected:
					Timer			( void 				);
					~Timer			( void 				);

public:
	static Timer* 	getInstance		( void 				);
	static void		freeInstance	( void 				);
	void 			init			( void 				);
	void 			update			( void 				);
	unsigned long 	getTimeMSec		( void 				);
	unsigned long 	getTime			( void 				);
	float 			getAnimTime		( void 				);
	float 			getFps			( void 				);
#ifdef _WIN32
	void			start			( DWORD&	time1 	);
	float			stop			( DWORD&	time1	);
#elif defined __unix
	void			start			( timeval& 	time1 	);
	float			stop			( timeval&	time1	);
#endif

private:
	unsigned long	currentTime;
	unsigned long	lastTime;
	static Timer*	singleton;
};

#endif

//=======================================================================================
