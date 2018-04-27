/******************************************************************************

 @File         MDKPrecisionTimer.cpp

 @Title        MDKTools

 @Copyright    Copyright (C) 2009 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  
 
******************************************************************************/

#include "MDKPrecisionTimer.h"
#include <stdarg.h>
#include <stdio.h>
#include <assert.h>

namespace MDK {
	namespace Common {

		

	/***********************************************************************
	 * Timing functions
	 ***********************************************************************/
	#ifdef _WIN32
		int gettimeofday(struct timeval *tv, struct timezone *tz) {
			if (NULL != tv) {
				LARGE_INTEGER lpPerformanceCount, Frequency;
				if (!QueryPerformanceCounter(&lpPerformanceCount) ||
					!QueryPerformanceFrequency(&Frequency))
					return -1;

				double sec = lpPerformanceCount.QuadPart / (double)Frequency.QuadPart;
				tv->tv_sec = (long)sec;
				tv->tv_usec = (long)((sec - tv->tv_sec) * 1000000.0);
			}
			return 0;
		}
	#elif defined (__APPLE__) || defined (__linux__) || defined (__SYMBIAN32__)
		// Don't do anything as the timer should be already in <sys/time.h>
	#else
		int gettimeofday(struct timeval *tv, struct timezone *tz)
		{
			if (g_shell)
			{
				long time = getShell()->PVRShellGetTime();
				tv->tv_sec = time / 1000UL;
				tv->tv_usec = (time % 1000UL) * 1000;
			}
			else
			{
				tv->tv_sec = 0UL;
				tv->tv_usec = 0UL;
			}
			return 0;
		}
	#endif // _WIN32


		/********************************************************
		 * PrecisionTimer class implementation
		 ********************************************************/
		PrecisionTimer::PrecisionTimer() {
			Start();
		}

		bool PrecisionTimer::Start() {
			if (gettimeofday(&m_sStartTime, NULL)) {
				return false;
			}
			m_sTimePrev = m_sStartTime;

			m_fDeltaTime = 0.0f;
			m_fTimePrev = 0.0f;
			return true;
		}

		bool PrecisionTimer::Update() {
			timeval sTime;
			if (gettimeofday(&sTime, NULL)) {
				return false;
			}
			m_fTimePrev = ToSeconds(sTime.tv_sec - m_sStartTime.tv_sec, sTime.tv_usec - m_sStartTime.tv_usec);
			m_fDeltaTime = ToSeconds(sTime.tv_sec - m_sTimePrev.tv_sec, sTime.tv_usec - m_sTimePrev.tv_usec);
			if (m_sTimePrev.tv_sec == sTime.tv_sec && m_sTimePrev.tv_usec == sTime.tv_usec)
				return false;
			m_sTimePrev = sTime;
			return true;
		}

	}
}
