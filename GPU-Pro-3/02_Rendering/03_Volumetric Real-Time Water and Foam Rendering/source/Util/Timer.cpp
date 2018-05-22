
#include "Timer.h"

#include <stdlib.h>


// -----------------------------------------------------------------------------
// -------------------------------- Timer::Timer -------------------------------
// -----------------------------------------------------------------------------
Timer::Timer(void)
{
    QueryPerformanceFrequency(&frequency);
    startCount.QuadPart = 0;
    endCount.QuadPart = 0;

    stopped = 0;
    startTimeInMicroSec = 0;
    endTimeInMicroSec = 0;
}

// -----------------------------------------------------------------------------
// ------------------------------- Timer::~Timer -------------------------------
// -----------------------------------------------------------------------------
Timer::~Timer(void)
{
}

// -----------------------------------------------------------------------------
// -------------------------------- Timer::Start -------------------------------
// -----------------------------------------------------------------------------
void Timer::Start(void)
{
    stopped = 0;
    QueryPerformanceCounter(&startCount);
}

// -----------------------------------------------------------------------------
// -------------------------------- Timer::Stop --------------------------------
// -----------------------------------------------------------------------------
void Timer::Stop(void)
{
    stopped = 1;
    QueryPerformanceCounter(&endCount);
}

// -----------------------------------------------------------------------------
// ----------------------- Timer::GetDeltaTimeInMicroSec -----------------------
// -----------------------------------------------------------------------------
double Timer::GetDeltaTimeInMicroSec(void)
{
    if(!stopped)
        QueryPerformanceCounter(&endCount);

    startTimeInMicroSec = startCount.QuadPart * (1000000.0 / frequency.QuadPart);
    endTimeInMicroSec = endCount.QuadPart * (1000000.0 / frequency.QuadPart);

    return endTimeInMicroSec - startTimeInMicroSec;
}

// -----------------------------------------------------------------------------
// ---------------------- Timer::getElapsedTimeInMilliSec ----------------------
// -----------------------------------------------------------------------------
double Timer::GetDeltaTimeInMilliSec(void)
{
    return this->GetDeltaTimeInMicroSec() * 0.001;
}

// -----------------------------------------------------------------------------
// -------------------------- Timer::GetDeltaTimeInSec -------------------------
// -----------------------------------------------------------------------------
double Timer::GetDeltaTimeInSec(void)
{
    return this->GetDeltaTimeInMicroSec() * 0.000001;
}

// -----------------------------------------------------------------------------
// ---------------------------- Timer::GetDeltaTime ----------------------------
// -----------------------------------------------------------------------------
double Timer::GetDeltaTime(void)
{
    return this->GetDeltaTimeInSec();
}
