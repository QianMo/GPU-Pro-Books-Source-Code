#pragma once


#include <essentials/types.h>

#ifdef MAXEST_FRAMEWORK_WINDOWS
	#include <Windows.h>
#else
	#include <sys/time.h>
#endif


using namespace NEssentials;


namespace NSystem
{
	uint64 TickCount();

	//

	inline uint64 TickCount()
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			LARGE_INTEGER frequency, counter;

			QueryPerformanceFrequency(&frequency);
			QueryPerformanceCounter(&counter);

			double time = (double)counter.QuadPart / ((double)frequency.QuadPart / 1000000.0);

			return (uint64)time;
		#else
			struct timeval tv;
			struct timezone tz;
			uint64_t sec;
			uint64_t usec;

			gettimeofday(&tv, &tz);

			sec = (uint64_t)tv.tv_sec;
			usec = (uint64_t)tv.tv_usec;

			return sec*1000000l + usec;
		#endif
	}
}
