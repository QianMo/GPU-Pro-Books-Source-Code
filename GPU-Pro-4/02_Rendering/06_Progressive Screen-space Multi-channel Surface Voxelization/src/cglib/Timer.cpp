/*-------------------- TIMER FUNCTIONS --------------------*/

#include "Timer.h"
#include "cglibdefines.h"

Timer::Timer(const char* name)
{
	times = 0;
	timer_name = std::string(name);
	timer_frames = 100;
#ifdef USE_TIMER_QUERY
	glGenQueries (1, queries);
	ellapsed_time = 0;
#else
	total_time = 0;
#endif
}

Timer::Timer(const char* name, int frames)
{
	times = 0;
	timer_name = std::string(name);
	timer_frames = frames;
#ifdef USE_TIMER_QUERY
	glGenQueries (1, queries);
	ellapsed_time = 0;
#else
	total_time = 0;
#endif
}

void Timer::start()
{
#ifdef USE_TIMER_QUERY
	if (times == 0)
		glBeginQuery (GL_TIME_ELAPSED_EXT, queries[0]);

#elif WIN32
	cur_time = timeGetTime();
#endif
}

void Timer::stop()
{
#ifdef USE_TIMER_QUERY
	if (times == 0)
	{
		glEndQuery (GL_TIME_ELAPSED_EXT);

		int available = 0;
		while (! available)
			glGetQueryObjectiv (queries[0], GL_QUERY_RESULT_AVAILABLE, &available);
		
		glGetQueryObjectui64vEXT (queries[0], GL_QUERY_RESULT, &ellapsed_time);
	}

	if (times == timer_frames)	// get the query result every 'timer_frames' frames
	{
	//	EAZD_PRINT (timer_name << ": " << queries[0] << " " << ellapsed_time / 1000000.0f << " ms");
		times = 0;
	}
	else
		times++;

#elif WIN32	
	total_time += timeGetTime() - cur_time;

	times++;

	if (times == timer_frames)
	{	
		EAZD_PRINT (timer_name << ": " << total_time / (times) << " ms");
		total_time = 0;
		times = 0;
	}
#endif
}
