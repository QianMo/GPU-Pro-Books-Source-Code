// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

#include "Performance.h"

#ifdef LINUX
#include <sys/time.h>
#endif

//------------------------------------------------
// Constructor
//------------------------------------------------
Performance::Performance(void)
{
	m_iFrameCount = 0;

	//Initialize timer counters
#ifdef WINDOWS
	QueryPerformanceCounter(&m_Timer);
	m_FPSTimer = m_Timer;
	QueryPerformanceFrequency(&m_Timer_Frequency);

	QueryPerformanceCounter(&m_TimerRender);
	QueryPerformanceFrequency(&m_TimerFrequencyRender);
#elif defined(LINUX)
	timeval t1;
	gettimeofday(&t1,NULL);
    m_FPSTimer = m_Timer =t1.tv_sec + (t1.tv_usec/1000000.0);

	timeval t2;
	gettimeofday(&t2,NULL);
    m_TimerRender =t2.tv_sec + (t2.tv_usec/1000000.0);
#endif
}

//------------------------------------------------
// Updates Frames per second (FPS)
//------------------------------------------------
bool Performance::updateFPS(char *aux, char *name)
{
	LARGE_INTEGER Timer;
#ifdef WINDOWS
	QueryPerformanceCounter(&Timer);
	m_iFrameCount++;
	float step = float(Timer.QuadPart-m_FPSTimer.QuadPart)/float(m_Timer_Frequency.QuadPart);
#elif defined(LINUX)
		timeval t1;
	gettimeofday(&t1,NULL);
    Timer = t1.tv_sec + (t1.tv_usec/1000000.0);
	m_iFrameCount++;
	float step = Timer-m_FPSTimer;
#endif

	if ( (m_iFrameCount>=100) || (step>=1.0) )
	{
		//sprintf_s(aux,1024,"FPS: %f (%f spf) with up to %d reflections", float(m_iFrameCount)/step,step/float(m_iFrameCount),m_Effect->GetNumReflexes()-1);
		//sprintf_s(aux,1024,"%s FPS: %lf (%lf spf), %d/%d threads", name, float(m_FrameCount)/step,step/float(m_FrameCount),m_iNumThreads,m_numCPU);
#ifdef WINDOWS
		sprintf_s(aux,1024,"%s FPS: %lf (%lf spf)", name, float(m_iFrameCount)/step,step/float(m_iFrameCount));
#elif defined(LINUX)
		sprintf(aux,"%s FPS: %lf (%lf spf)", name, float(m_iFrameCount)/step,step/float(m_iFrameCount));
#endif
		m_iFrameCount = 0;
		m_FPSTimer = Timer;
		return true;
	}
	return false;
}

//------------------------------------------------
// Function to meassure global time (used for
// animation).
//------------------------------------------------
float Performance::updateTime( void )
{
	LARGE_INTEGER Timer;
#ifdef WINDOWS
	QueryPerformanceCounter(&Timer);
	float step = float(Timer.QuadPart-m_Timer.QuadPart)/float(m_Timer_Frequency.QuadPart);
#elif defined(LINUX)
	timeval t1;
	gettimeofday(&t1,NULL);
    Timer = t1.tv_sec + (t1.tv_usec/1000000.0);
	float step = Timer - m_Timer;
#endif
	m_Timer=Timer;
	return step;
}

//------------------------------------------------
// Function to meassure the time neccesary to
// render a single frame
//------------------------------------------------
float Performance::updateTimeRender( void )
{
	LARGE_INTEGER Timer;
#ifdef WINDOWS
	QueryPerformanceCounter(&Timer);
	float step = float(Timer.QuadPart-m_TimerRender.QuadPart)/float(m_TimerFrequencyRender.QuadPart);
#elif defined(LINUX)
	timeval t1;
	gettimeofday(&t1,NULL);
    Timer = t1.tv_sec + (t1.tv_usec/1000000.0);
	float step = Timer - m_TimerRender;
#endif
	m_TimerRender=Timer;
	return step;
}

//------------------------------------------------
// Destructor
//------------------------------------------------
Performance::~Performance(void)
{
}
