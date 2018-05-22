
#ifndef __TIMER__H__
#define __TIMER__H__

#include <windows.h>


// -----------------------------------------------------------------------------
/// 
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class Timer
{
public:
	Timer();
	~Timer();

	/// Start the timer
	void Start(void);
	/// Stop the timer
	void Stop(void);

	/// Get delta time
	double GetDeltaTime(void);
	/// Get delta time in seconds
	double GetDeltaTimeInSec(void);
	/// Get delta time in milliseconds
	double GetDeltaTimeInMilliSec(void);
	/// Get delta time in microseconds
	double GetDeltaTimeInMicroSec(void);

private:
	double startTimeInMicroSec;	// starting time in micro-second
	double endTimeInMicroSec;	// ending time in micro-second
	int stopped;

	LARGE_INTEGER frequency;
	LARGE_INTEGER startCount;
	LARGE_INTEGER endCount;
};

#endif
