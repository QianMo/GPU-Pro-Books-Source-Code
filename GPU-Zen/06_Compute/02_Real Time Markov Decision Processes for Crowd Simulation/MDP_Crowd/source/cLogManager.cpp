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
#include "cLogManager.h"

//=======================================================================================
//
LogManager::LogManager( string& log_filename )
{
    timeBuf                             = new char[ 256 ];
	filename						    = log_filename;
	sections_map[LERROR]				= string( "___ERROR" 	);
	sections_map[GL_ERROR]			    = string( "__GL_ERR" 	);
	sections_map[WARNING]			    = string( "_WARNING" 	);
	sections_map[CONFIGURATION]		    = string( "__CONFIG" 	);
	sections_map[XML]				    = string( "_____XML" 	);
	sections_map[CONTEXT]			    = string( "_CONTEXT" 	);
	sections_map[GLSL_MANAGER]	    	= string( "____GLSL" 	);
	sections_map[FBO_MANAGER]		    = string( "_____FBO" 	);
	sections_map[VBO_MANAGER]			= string( "_____VBO" 	);
	sections_map[EXTENSION_MANAGER]	    = string( "_EXT_MAN" 	);
	sections_map[TEXTURE_MANAGER]	    = string( "_TEXTURE" 	);
	sections_map[CROWD_MANAGER]		    = string( "___CROWD" 	);
	sections_map[OBSTACLE_MANAGER]	    = string( "OBSTACLE" 	);
	sections_map[BOUNDING_VOLUME]	    = string( "_BOUND_V" 	);
	sections_map[MODEL]				    = string( "___MODEL" 	);
	sections_map[GPU_RAM]			    = string( "_GPU_RAM" 	);
	sections_map[ALUT]				    = string( "____ALUT" 	);
	sections_map[FMOD]				    = string( "____FMOD" 	);
	sections_map[OGG]				    = string( "_____OGG" 	);
	sections_map[MD2]				    = string( "_____MD2" 	);
	sections_map[SKYBOX]			    = string( "__SKYBOX" 	);
	sections_map[INFORMATION]		    = string( "____INFO" 	);
	sections_map[STATUS]			    = string( "__STATUS" 	);
	sections_map[STATISTICS]		    = string( "___STATS" 	);
	sections_map[CLEANUP]			    = string( "_CLEANUP" 	);
	sections_map[NET]				    = string( "_____NET" 	);
	sections_map[EXIT]				    = string( "____EXIT" 	);
	sections_map[STATIC_LOD]		    = string( "___S_LOD" 	);
	sections_map[CUDA]				    = string( "____CUDA" 	);
	sections_map[MDP]				    = string( "_____MDP" 	);

	html_colors_map[LERROR]				= string( "#FF0000"		);
	html_colors_map[GL_ERROR]			= string( "#FF0000" 	);
	html_colors_map[WARNING]			= string( "#FF8000" 	);
	html_colors_map[CONFIGURATION]		= string( "#610B38" 	);
	html_colors_map[XML]				= string( "#380B61" 	);
	html_colors_map[CONTEXT]			= string( "#0404B4" 	);
	html_colors_map[GLSL_MANAGER]		= string( "#5E9D32" 	);
	html_colors_map[FBO_MANAGER]		= string( "#5F04B4" 	);
	html_colors_map[VBO_MANAGER]		= string( "#95D17B" 	);
	html_colors_map[EXTENSION_MANAGER]	= string( "#AEB404" 	);
	html_colors_map[TEXTURE_MANAGER]	= string( "#DF7401" 	);
	html_colors_map[CROWD_MANAGER]		= string( "#DA4422" 	);
	html_colors_map[OBSTACLE_MANAGER]	= string( "#6543DC" 	);
	html_colors_map[BOUNDING_VOLUME]	= string( "#9A2EFE" 	);
	html_colors_map[MODEL]				= string( "#2E2EFE" 	);
	html_colors_map[GPU_RAM]			= string( "#2E9AFE" 	);
	html_colors_map[ALUT]				= string( "#58FAAC" 	);
	html_colors_map[FMOD]				= string( "#58FA58" 	);
	html_colors_map[OGG]				= string( "#393B0B" 	);
	html_colors_map[MD2]				= string( "#F78181" 	);
	html_colors_map[SKYBOX]				= string( "#0B6138" 	);
	html_colors_map[INFORMATION]		= string( "#000000" 	);
	html_colors_map[STATUS]				= string( "#000000" 	);
	html_colors_map[STATISTICS]			= string( "#888888" 	);
	html_colors_map[CLEANUP]			= string( "#FF0000" 	);
	html_colors_map[NET]				= string( "#5AA250" 	);
	html_colors_map[EXIT]				= string( "#FF0000" 	);
	html_colors_map[STATIC_LOD]			= string( "#385B66" 	);
	html_colors_map[CUDA]				= string( "#009900" 	);
	html_colors_map[MDP]				= string( "#FEDB39" 	);

    time_t currTime;
    struct tm *localTime;
    currTime = time( NULL );
    localTime = localtime( &currTime );
	printf( "SRL-BHA::CASIM_Init@%s\n", asctime( localTime ) );
	cout << "Log File: " << filename.c_str() << endl;
	logFile.open( filename.c_str(), ios::out | ios::trunc );
	logFile << "<html>\n";
	logFile << "<table border=\"1\">\n";
	logFile << "<tr>\n";
	logFile << "<td colspan=\"3\" align=\"center\">\n";
	logFile << "<b>SRL-BHA::CASIM_Init@<span style=\"color: #0000FF;\">";
	logFile << asctime( localTime ) << "</span></b>\n";
	logFile << "</td>\n";
	logFile << "</tr>\n";
	separator();
}
//
//=======================================================================================
//
LogManager::~LogManager( void )
{
	logFile << "</table>\n";
	logFile << "</html>\n";
	filename.erase();
	sections_map.clear();
	logFile.close();
}
//
//=======================================================================================
//
void LogManager::log( int section, const char* format, ... )
{
    va_list args;
    char buffer[4096];

	getTime();
	cout << timeBuf << " | " << sections_map[section].c_str() << " | ";

    va_start( args, format );
	vsprintf( buffer, format, args );
    va_end( args );

	cout << buffer << endl;

	string sbuffert( buffer );
	string sbuffer;
	for( unsigned int b = 0; b < sbuffert.length(); b++ )
	{
		if( sbuffert[b] == '\n' )
		{
			sbuffer.append( string("<br>") );
		}
		else
		{
			sbuffer.push_back( sbuffert[b] );
		}
	}

	logFile << "<tr>\n";
	logFile << "<td><span style=\"color: #0000FF;\">" << timeBuf << "</span></td><td>";
	logFile << "<span style=\"color: " << html_colors_map[section].c_str() << "\">";
	logFile << "<b>" << sections_map[section].c_str() << "</b></span></td><td>";
	logFile << sbuffer.c_str() << "</td>\n";
	logFile << "</tr>\n";
}
//
//=======================================================================================
//
void LogManager::separator( void )
{
	file_separator();
	console_separator();
}
//
//=======================================================================================
//
void LogManager::console_log( int section, const char* format, ... )
{
    va_list args;
    char buffer[1024];
	getTime();
	cout << timeBuf << " | " << sections_map[section].c_str() << " | ";
    va_start( args, format );
	vsprintf( buffer, format, args );
    va_end( args );
	cout << buffer << endl;
}
//
//=======================================================================================
//
void LogManager::file_log( int section, const char* format, ... )
{
    va_list args;
    char buffer[1024];
    va_start( args, format );
	vsprintf( buffer, format, args );
    va_end( args );

    getTime();
	logFile << "<tr>\n";
	logFile << "<td><span style=\"color: #0000FF;\">" << timeBuf << "</span></td><td>";
	logFile << "<span style=\"color: " << html_colors_map[section].c_str() << "\">";
	logFile << "<b>" << sections_map[section].c_str() << "</b></span></td><td>";
	logFile << buffer << "</td>\n";
	logFile << "</tr>\n";
}
//
//=======================================================================================
//
void LogManager::console_separator( void )
{
	cout << "---------------+----------+-----------------------------";
	cout << "-------------------------" << endl;
}
//
//=======================================================================================
//
void LogManager::file_separator( void )
{
	logFile << "<tr><td align=\"center\"><span style=\"color: #CCCCCC\">TIME</span>";
	logFile << "</td><td align=\"center\"><span style=\"color: #CCCCCC\">";
	logFile << "SECTION</span></td>";
	logFile << "<td align=\"center\"><span style=\"color: #CCCCCC\">";
	logFile << "MESSAGE</span></td></tr>\n";
}
//
//=======================================================================================
//
void LogManager::getTime( void )
{
#if defined _WIN32
	SYSTEMTIME st;
    GetLocalTime( &st );
	printTime( "%02d:%02d:%02d:%05d", 
			   st.wHour, 
			   st.wMinute, 
			   st.wSecond, 
			   st.wMilliseconds );
#elif defined _UNIX	
	time_t t = time( NULL );
	struct tm tm = *localtime( &t );
	struct timeval lT;
	gettimeofday( &lT, NULL );
	int ms = (int)(lT.tv_usec / 1000.0);
	printTime( "%02d:%02d:%02d:%05d",
			   tm.tm_hour,
			   tm.tm_min,
			   tm.tm_sec,
			   ms                   );
#endif
}
//
//=======================================================================================
//
void LogManager::printTime( const char* format, ... )
{
	va_list args;
    va_start( args, format );
	vsprintf( timeBuf, format, args );
	//perror( timeBuf );
    va_end( args );
}
//
//=======================================================================================
//
void LogManager::console_prc( int curr_val, int max_val )
{
	float full = (float)max_val;
	float prc  = (float)curr_val * 100.0f / full;
	cout.precision( 4 );
	if( prc <= 10.0f )
	{
		cout << "\r";
		cout << "[|         ] " << prc << "% ";
	}
	else if( prc <= 20.0f )
	{
		cout << "\r";
		cout << "[||        ] " << prc << "% ";
	}
	else if( prc <= 30.0f )
	{
		cout << "\r";
		cout << "[|||       ] " << prc << "% ";
	}
	else if( prc <= 40.0f )
	{
		cout << "\r";
		cout << "[||||      ] " << prc << "% ";
	}
	else if( prc <= 50.0f )
	{
		cout << "\r";
		cout << "[|||||     ] " << prc << "% ";
	}
	else if( prc <= 60.0f )
	{
		cout << "\r";
		cout << "[||||||    ] " << prc << "% ";
	}
	else if( prc <= 70.0f )
	{
		cout << "\r";
		cout << "[|||||||   ] " << prc << "% ";
	}
	else if( prc <= 80.0f )
	{
		cout << "\r";
		cout << "[||||||||  ] " << prc << "% ";
	}
	else if( prc <= 90.0f )
	{
		cout << "\r";
		cout << "[||||||||| ] " << prc << "% ";
	}
	else
	{
		cout << "\r";
		cout << "[||||||||||] " << prc << "% ";
	}
}
//
//=======================================================================================
//
void LogManager::logStatistics( unsigned int tex_w,
								unsigned int vert_c,
								unsigned int frame_vert_c,
								unsigned int vert_size
							  )
{
	log( STATISTICS,
		 "Textures weight: %i MB",
		 BYTE2MB( tex_w ) );
	log( STATISTICS,
		 "Vertices in scene: %i K (%i MB)",
		 vert_c / 1000,
		 BYTE2MB( (vert_c * vert_size) ) );
	log( STATISTICS,
		 "Vertices per frame: %i K (%i MB)",
		 frame_vert_c / 1000,
		 BYTE2MB( (frame_vert_c * vert_size) ) );
	log( STATISTICS,
		 "Triangles per frame: %i K (%i MB)",
		 frame_vert_c / 3000,
		 BYTE2MB( (frame_vert_c * vert_size) ) );
}
//
//=======================================================================================
//
void LogManager::logPeakFps( float			peakFps,
							 float			avgFps,
							 unsigned long	frame_c,
							 float			spf,
							 unsigned int	culled,
							 unsigned int	total
						   )
{
	log( STATISTICS,
		 "New PEAK FPS: %.2f. Average FPS: %.2f. Frame: %u.",
		 peakFps,
		 avgFps,
		 frame_c );
	log( STATISTICS,
		"Seconds per frame: %.3f. Culled objects: %i (%i).",
		 spf,
		 culled,
		 total );
}
//
//=======================================================================================
//
void LogManager::logLowFps( float			lowFps,
						    float			avgFps,
							unsigned long	frame_c,
							float			spf,
							unsigned int	culled,
							unsigned int	total
						  )
{
	log( STATISTICS,
		 "New LOW  FPS: %.2f. Average FPS: %.2f. Frame: %u.",
		 lowFps,
		 avgFps,
		 frame_c );
	log( STATISTICS,
		"Seconds per frame: %.3f. Culled objects: %i.",
		 spf,
		 culled,
		 total );
}
//
//=======================================================================================
//
void LogManager::file_prc( int section, int curr_val, int max_val )
{
	getTime();
	float full = (float)max_val;
	float prc  = (float)curr_val * 100.0f / full;
	logFile << "<tr>\n";
	logFile << "<td><span style=\"color: #0000FF;\">" << timeBuf << "</span></td><td>";
	logFile << "<span style=\"color: " << html_colors_map[section].c_str() << "\">";
	logFile << sections_map[section].c_str() << "</span></td><td>";

	if( prc <= 10.0f )
	{
		logFile << "[";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #DDDDDD;\">| | | | | | | | |</span>";
		logFile << "] ";
		logFile.precision( 4 );
		logFile << prc << "% ";
	}
	else if( prc <= 20.0f )
	{
		logFile << "[";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #DDDDDD;\">| | | | | | | |</span>";
		logFile << "] ";
		logFile.precision( 4 );
		logFile << prc << "% ";
	}
	else if( prc <= 30.0f )
	{
		logFile << "[";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #DDDDDD;\">| | | | | | |</span>";
		logFile << "] ";
		logFile.precision( 4 );
		logFile << prc << "% ";
	}
	else if( prc <= 40.0f )
	{
		logFile << "[";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #DDDDDD;\">| | | | |</span>";
		logFile << "] ";
		logFile.precision( 4 );
		logFile << prc << "% ";
	}
	else if( prc <= 50.0f )
	{
		logFile << "[";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #F88017;\">| </span>";
		logFile << "<span style=\"color: #DDDDDD;\">| | | | |</span>";
		logFile << "] ";
		logFile.precision( 4 );
		logFile << prc << "% ";
	}
	else if( prc <= 60.0f )
	{
		logFile << "[";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #F88017;\">| </span>";
		logFile << "<span style=\"color: #F88017;\">| </span>";
		logFile << "<span style=\"color: #DDDDDD;\">| | | |</span>";
		logFile << "] ";
		logFile.precision( 4 );
		logFile << prc << "% ";
	}
	else if( prc <= 70.0f )
	{
		logFile << "[";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #F88017;\">| </span>";
		logFile << "<span style=\"color: #F88017;\">| </span>";
		logFile << "<span style=\"color: #F62817;\">| </span>";
		logFile << "<span style=\"color: #DDDDDD;\">| | |</span>";
		logFile << "] ";
		logFile.precision( 4 );
		logFile << prc << "% ";
	}
	else if( prc <= 80.0f )
	{
		logFile << "[";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #F88017;\">| </span>";
		logFile << "<span style=\"color: #F88017;\">| </span>";
		logFile << "<span style=\"color: #F62817;\">| </span>";
		logFile << "<span style=\"color: #F62817;\">| </span>";
		logFile << "<span style=\"color: #DDDDDD;\">| |</span>";
		logFile << "] ";
		logFile.precision( 4 );
		logFile << prc << "% ";
	}
	else if( prc <= 90.0f )
	{
		logFile << "[";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #F88017;\">| </span>";
		logFile << "<span style=\"color: #F88017;\">| </span>";
		logFile << "<span style=\"color: #F62817;\">| </span>";
		logFile << "<span style=\"color: #F62817;\">| </span>";
		logFile << "<span style=\"color: #FF0000;\">| </span>";
		logFile << "<span style=\"color: #DDDDDD;\">| </span>";
		logFile << "] ";
		logFile.precision( 4 );
		logFile << prc << "% ";
	}
	else
	{
		logFile << "[";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #00FF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #FFFF00;\">| </span>";
		logFile << "<span style=\"color: #F88017;\">| </span>";
		logFile << "<span style=\"color: #F88017;\">| </span>";
		logFile << "<span style=\"color: #F62817;\">| </span>";
		logFile << "<span style=\"color: #F62817;\">| </span>";
		logFile << "<span style=\"color: #FF0000;\">| </span>";
		logFile << "<span style=\"color: #FF0000;\">| </span>";
		logFile << "] ";
		logFile.precision( 4 );
		logFile << prc << "% ";
	}
	logFile << "</td></tr>";
}
//
//=======================================================================================
