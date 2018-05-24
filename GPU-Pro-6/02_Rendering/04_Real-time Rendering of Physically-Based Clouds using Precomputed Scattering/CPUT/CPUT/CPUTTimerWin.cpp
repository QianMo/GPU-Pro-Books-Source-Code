//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------
#include "CPUTTimerWin.h"

//
// Timer is initially not running and set to a zero state.
// Performance counter frequency is obtained.
//
CPUTTimerWin::CPUTTimerWin():mbCounting(false)
{
    ResetTimer();

	//
	// Frequency only needs to be collected once.
	//
	QueryPerformanceFrequency(&mlFrequency);
}

//
// Reset timer to zero
//
void CPUTTimerWin::ResetTimer()
{
    mlStartCount.QuadPart   = 0;
    mlLastMeasured.QuadPart = 0;
    mbCounting              = false;
}

//
// Starts timer and resets everything. If timer is already running,
// nothing happens.
//
void CPUTTimerWin::StartTimer()
{
    if(!mbCounting)
    {
        ResetTimer();
        mbCounting = true;
        QueryPerformanceCounter(&mlLastMeasured);
        mlStartCount = mlLastMeasured;
    }
}

//
// If the timer is running, stops the timer and returns the time in seconds since StartTimer() was called.
//
// If the timer is not running, returns the time in seconds between the
// last start/stop pair.
//
double CPUTTimerWin::StopTimer()
{
    if(mbCounting)
    {
		mbCounting = false;
        QueryPerformanceCounter(&mlLastMeasured);
	}

	return (((double)(mlLastMeasured.QuadPart - mlStartCount.QuadPart)) / ((double)(mlFrequency.QuadPart)));
}

//
// If the timer is running, returns the total time in seconds since StartTimer was called.
//
// If the timer is not running, returns the time in seconds between the
// last start/stop pair.
//
double CPUTTimerWin::GetTotalTime()
{
	LARGE_INTEGER temp;

	if (mbCounting) {
		QueryPerformanceCounter(&temp);
		return ((double)(temp.QuadPart - mlStartCount.QuadPart) / (double)(mlFrequency.QuadPart));
	}

	return ((double)(mlLastMeasured.QuadPart - mlStartCount.QuadPart) / (double)(mlFrequency.QuadPart));
}

//
// If the timer is running, returns the total time in seconds that the timer
// has run since it was last started or elapsed time was read from. Updates last measured time.
//
// If the timer is not running, returns 0.
//
double CPUTTimerWin::GetElapsedTime()
{
	LARGE_INTEGER temp;
	LARGE_INTEGER elapsedTime;
	elapsedTime.QuadPart = 0;

	if (mbCounting) {
		QueryPerformanceCounter(&temp);
		elapsedTime.QuadPart = temp.QuadPart - mlLastMeasured.QuadPart;
		mlLastMeasured = temp;
	}

	return ((double)elapsedTime.QuadPart / (double)mlFrequency.QuadPart);
}
