#pragma once

#define USE_TIMER_QUERY

#ifdef WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
	#ifndef USE_TIMER_QUERY
		#include <Mmsystem.h>
	#endif
#endif

#include <string>

#include <GL/glew.h>

class Timer
{
#ifdef USE_TIMER_QUERY
	GLuint64EXT ellapsed_time;
	GLuint queries[1];
#elif WIN32
	DWORD cur_time;
	DWORD total_time;
#else
	double cur_time;
	double total_time;
#endif
	unsigned int times;
	std::string timer_name;
	int timer_frames;

public:
	Timer(const char* name);
	Timer(const char* name, int frames);
	void start();
	void stop();

	std::string getName () { return timer_name; }
	double getMsTime () { return (double) ellapsed_time / 1000000.0f; }
};
