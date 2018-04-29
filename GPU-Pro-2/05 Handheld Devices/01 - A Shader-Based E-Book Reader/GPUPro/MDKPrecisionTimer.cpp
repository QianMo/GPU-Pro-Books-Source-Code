/******************************************************************************

 @File         MDKPrecisionTimer.cpp

 @Title        MDKTools

 @Copyright    Copyright (C) 2010 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  Timer class
 
******************************************************************************/

#include "MDKMisc.h"
#include "MDKPrecisionTimer.h"


/***********************************************************************
* Timing functions
***********************************************************************/
#if defined( _WIN32 ) && !defined( __BADA__ )
#include <windows.h>


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
#elif defined ( __BADA__ )
#include <FSystem.h>

int gettimeofday( struct timeval *tv, struct timezone *tz )
{
	unsigned int t = Timer::GetShell()->PVRShellGetTime();

	tv->tv_sec = t / 1000;
	tv->tv_usec = (t % 1000) * 1000;

	return 0;
}
#elif defined (__APPLE__) || defined (__linux__) || defined (__SYMBIAN32__) || defined (ANDROID)
// Don't do anything as the timer should be already in <sys/time.h>
#else
int gettimeofday(struct timeval *tv, struct timezone *tz)
{
	tv->tv_sec = 0UL;
	tv->tv_usec = 0UL;
	return 0;
}
#endif // _WIN32


/********************************************************
 * PrecisionTimer class implementation
 ********************************************************/
PrecisionTimer::PrecisionTimer() {
	Start();
}

bool PrecisionTimer::Start()
{
	if (gettimeofday(&m_sStartTime, NULL))
	{
		return false;
	}
	m_sTimePrev = m_sStartTime;

	m_fDeltaTime = 0.0f;
	m_fTimePrev = 0.0f;
	return true;
}

bool PrecisionTimer::Update()
{
	timeval sTime;
	if( gettimeofday( &sTime, NULL ) )
	{
		return false;
	}
	m_fTimePrev = ToSeconds(sTime.tv_sec - m_sStartTime.tv_sec, sTime.tv_usec - m_sStartTime.tv_usec);
	m_fDeltaTime = ToSeconds(sTime.tv_sec - m_sTimePrev.tv_sec, sTime.tv_usec - m_sTimePrev.tv_usec);

	if (m_sTimePrev.tv_sec == sTime.tv_sec && m_sTimePrev.tv_usec == sTime.tv_usec)
	{
		return false;
	}

	m_sTimePrev = sTime;

	return true;
}


/********************************************************
 * FixedTimer class implementation
 ********************************************************/


