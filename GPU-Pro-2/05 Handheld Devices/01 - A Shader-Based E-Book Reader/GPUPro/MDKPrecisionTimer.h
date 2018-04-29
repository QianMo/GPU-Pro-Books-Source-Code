/******************************************************************************

 @File         MDKPrecisionTimer.h

 @Title        MDKTools

 @Copyright    Copyright (C) 2010 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  Timer class
 
******************************************************************************/

#ifndef _MDK_PRECISION_TIMER_H_
#define _MDK_PRECISION_TIMER_H_

#if defined(_WIN32) && !defined(__BADA__)
	#include <time.h>
	#include <winsock.h>
#endif



/***********************************************************************
 * Timing functions
 ***********************************************************************/
#if defined(_WIN32) && !defined(__BADA__)

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

#elif defined (__APPLE__) || defined (__linux__) || defined (__SYMBIAN32__) || defined (ANDROID) 
	#include <sys/time.h>
#else
	typedef struct timeval {
		long tv_sec;
		long tv_usec;
	} timeval;

	int gettimeofday(struct timeval *tv, struct timezone *tz);
#endif // _WIN32

typedef float mdk_time;

//! Timer classes
/* Timer is an abstract class providing a timer interface
 */

class Timer
{

public:
	Timer() { }
	virtual ~Timer() { }

	//! Starts the timer.
	virtual bool Start() = 0;
	//! Update the timer.
	virtual bool Update() = 0;
	//! Restart the timer.
	virtual void Restart() { Start(); }

	virtual unsigned long GetTime() = 0;
	virtual mdk_time GetTimef() = 0;
	virtual mdk_time GetDeltaTimef() = 0;

	virtual void SetTimeStep(mdk_time step) = 0;

	static inline mdk_time ToSeconds(long sec, long usec) { return (mdk_time)sec + (mdk_time)usec / 1000000.0f; }


};

//! Class implementing a high precision timer (real, time-based)
class PrecisionTimer : public Timer
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
	virtual bool Start();

	//! Update the timer.
	virtual bool Update();

	//! Restart the timer.
	virtual void Restart(){ Start(); }

	//! Gets the number of milliseconds that have elapsed since the timer started.
	/*!

	*/
	unsigned long GetTime() { return (unsigned long)(m_fTimePrev * 1000.0); }

	//! Gets the number of seconds that have elapsed since the timer started, as a floating point value.
	virtual mdk_time GetTimef() { return m_fTimePrev; }

	//! Gets the number of seconds that have elapsed since the last time PrecisionTimer::update was called, as a floating point value.
	virtual mdk_time GetDeltaTimef() { return m_fDeltaTime; }

	//void offset(float seconds) { m_startTime = m_startTime-(seconds*1000); }
	//! 
	virtual void SetTimeStep(mdk_time step) { }

};

#define TIMER_DEFAULT_TIMESTEP (1.0f / 30.0f)

//! Class implementing a constant-step timer
class FixedTimer : public Timer
{
	float fTimeStep;
	float fTime;
public:
	FixedTimer() : fTimeStep(TIMER_DEFAULT_TIMESTEP), fTime(0.0f) { }
	FixedTimer(mdk_time step) : fTimeStep(step), fTime(0.0f) { }

	//! Starts the timer.
	virtual bool Start() { fTime = 0.0f; return true; }
	//! Update the timer.
	virtual bool Update() { fTime += fTimeStep; return true; }
	//! Restart the timer.
	virtual void Restart() { Start(); }

	virtual unsigned long GetTime() { return (unsigned long)(fTime * 1000.0f); }
	virtual mdk_time GetTimef() { return fTime; }
	virtual mdk_time GetDeltaTimef() { return fTimeStep; }

	virtual void SetTimeStep(mdk_time step) { fTimeStep = step; }
};


#endif //_MDK_PRECISION_TIMER_H_
