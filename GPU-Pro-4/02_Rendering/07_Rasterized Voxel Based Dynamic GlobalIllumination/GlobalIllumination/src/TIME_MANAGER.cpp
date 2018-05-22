#include <stdafx.h>
#include <TIME_MANAGER.h>

void TIME_MANAGER::Init()
{
	LONGLONG frequency;
	if(QueryPerformanceFrequency((LARGE_INTEGER*)&frequency)) 
	{
		performanceCounterAvailable = true;
		timeScale = 1.0/frequency;
	}
	else
		performanceCounterAvailable = false;
}

void TIME_MANAGER::Update()
{
	double tickCount;
	
	// use performance counter for timing, if available
	// ->otherwise use timeGetTime() 
	if(performanceCounterAvailable)
	{
		LONGLONG currentTime;
		QueryPerformanceCounter((LARGE_INTEGER*)&currentTime);
		tickCount = currentTime*timeScale*1000.0;
	}
	else
		tickCount = timeGetTime();	 
  
	// calculate fps
	static int fpsFrames = 0;
	static float fpsLastTime = 0.0f;
	float fpsTime = (float)tickCount*0.001f;
	fpsFrames++;
	if(fpsTime-fpsLastTime>1.0f)
	{
		fps = fpsFrames/(fpsTime-fpsLastTime);
		fpsLastTime = fpsTime;
		fpsFrames = 0;
	}

	// calculate frame-interval
	static double lastTickCount = 0.0;
	frameInterval = tickCount-lastTickCount;
	lastTickCount = tickCount;
}


