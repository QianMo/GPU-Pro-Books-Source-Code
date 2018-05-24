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
#include <time.h>
#ifdef __unix__
	#include <sys/time.h>
#endif
#include <vector>
#include <string>
#include <stdio.h>
#include <stdarg.h>
#include <iostream>
#include <fstream>
#include <map>

#include "cMacros.h"

using namespace std;

//=======================================================================================

#ifndef __LOG_MANAGER
#define __LOG_MANAGER

class LogManager
{
public:
								LogManager			( string& log_filename			);
								~LogManager			( void							);
public:
	enum					    LOG_MESSAGE_TYPE
								{
									LERROR,
									GL_ERROR,
									WARNING,
									CONFIGURATION,
									XML,
									CONTEXT,
									GLSL_MANAGER,
									FBO_MANAGER,
									VBO_MANAGER,
									EXTENSION_MANAGER,
									TEXTURE_MANAGER,
									CROWD_MANAGER,
									OBSTACLE_MANAGER,
									BOUNDING_VOLUME,
									MODEL,
									GPU_RAM,
									FMOD,
									ALUT,
									OGG,
									MD2,
									SKYBOX,
									INFORMATION,
									STATUS,
									STATISTICS,
									CLEANUP,
									NET,
									EXIT,
									STATIC_LOD,
									CUDA,
									MDP
								};
public:
	void						log					( int			section,
													  string		data			);
	void						log					( int			section,
													  const char*	format, ...		);
	void						console_log			( int			section,
													  const char*	format, ...		);
	void						file_log			( int			section,
													  const char*	format, ...		);
	void						separator			( void							);
	void						console_separator	( void							);
	void						file_separator		( void							);
	void						file_prc			( int			section,
													  int			curr_val,
													  int			max_val			);
	void						console_prc			( int			curr_val,
													  int			max_val			);
	void						logStatistics		( unsigned int	tex_w,
													  unsigned int	vert_c,
													  unsigned int	frame_vert_c,
													  unsigned int	vert_size		);
	void						logPeakFps			( float			peakFps,
													  float			avgFps,
													  unsigned long frame_c,
													  float			spf,
													  unsigned int	culled,
													  unsigned int	total			);
	void						logLowFps			( float			lowFps,
													  float			avgFps,
													  unsigned long frame_c,
													  float			spf,
													  unsigned int	culled,
													  unsigned int	total			);
private:
	map<int, string>			sections_map;
	map<int, string>			html_colors_map;
	ofstream					logFile;
	char*						timeBuf;
	string						filename;
private:
	void						getTime				( void							);
	void						printTime			( const char*	format, ...		);
};

#endif

//=======================================================================================
