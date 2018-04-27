/******************************************************************************

 @File         MDKPrecisionTimer.h

 @Title        MDKTools

 @Copyright    Copyright (C) 2009 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  
 
******************************************************************************/

#ifndef _MDK_PRECISION_TIMER_H_
#define _MDK_PRECISION_TIMER_H_

#include "PVRShell.h"

namespace MDK {
	namespace Common {

		/***********************************************************************
		 * Timing functions
		 ***********************************************************************/
		#ifdef _WIN32

			#include <windows.h>
			#include <winsock.h>
			#include <time.h>

			#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
			  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
			#else
			  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
			#endif
			 
			struct timezone 
			{
			  int  tz_minuteswest; // minutes W of Greenwich
			  int  tz_dsttime;     // type of dst correction
			};
			 
			int gettimeofday(struct timeval *tv, struct timezone *tz);

		#elif defined (__APPLE__) || defined (__linux__) || defined (__SYMBIAN32__)
			#include <sys/time.h>
		#else
			#error "-------------------------> Shell doesn't know the current OS. High resolution timer not available. <--------------------------"
			int gettimeofday(struct timeval *tv, struct timezone *tz);
		#endif // _WIN32

		typedef float mdk_time;

		//! Class implementing a high precision timer.
		class PrecisionTimer
		{
		private:
			struct timeval m_sStartTime;
			struct timeval m_sTimePrev;
			mdk_time m_fTimePrev;
			mdk_time m_fDeltaTime;
		public:
			//! Constructor
			PrecisionTimer();

			//! Starts the timer.
			bool Start();

			//! Update the timer.
			bool Update();

			//! Restart the timer.
			void Restart(){ Start(); }

			//! Gets the number of milliseconds that have elapsed since the timer started.
			/*!

			*/
			unsigned long GetTime() { return (unsigned long)(m_fTimePrev * 1000.0); }

			//! Gets the number of seconds that have elapsed since the timer started, as a floating point value.
			mdk_time GetTimef() { return m_fTimePrev; }

			//! Gets the number of seconds that have elapsed since the last time PrecisionTimer::update was called, as a floating point value.
			mdk_time GetDeltaTimef() { return m_fDeltaTime; }

			//void offset(float seconds) { m_startTime = m_startTime-(seconds*1000); }

			static inline mdk_time ToSeconds(long sec, long usec) { return (mdk_time)sec + (mdk_time)usec / 1000000.0f; }
		};

	}
}


#endif //_MDK_PRECISION_TIMER_H_
