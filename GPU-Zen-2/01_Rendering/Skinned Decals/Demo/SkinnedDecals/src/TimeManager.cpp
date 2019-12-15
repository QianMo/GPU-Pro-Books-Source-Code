#include <stdafx.h>
#include <TimeManager.h>

void TimeManager::Init()
{
  LONGLONG frequency;
  if(QueryPerformanceFrequency((LARGE_INTEGER*)&frequency)) 
  {
    performanceCounterAvailable = true;
    timeScale = 1.0 / frequency;
  }
  else
    performanceCounterAvailable = false;
}

void TimeManager::Update()
{
  const double tickCount = GetTickCount();
  
  // calculate fps
  static int fpsFrames = 0;
  static float fpsLastTime = 0.0f;
  float fpsTime = static_cast<float>(tickCount) * 0.001f;
  fpsFrames++;
  if((fpsTime - fpsLastTime) > 1.0f)
  {
    fps = fpsFrames / (fpsTime - fpsLastTime);
    fpsLastTime = fpsTime;
    fpsFrames = 0;
  }

  // calculate frame-interval
  static double lastTickCount = 0.0;
  frameInterval = tickCount-lastTickCount;
  lastTickCount = tickCount;
}

double TimeManager::GetTickCount() const
{
  double tickCount;

  if(performanceCounterAvailable)
  {
    LONGLONG currentTime;
    QueryPerformanceCounter((LARGE_INTEGER*)&currentTime);
    tickCount = currentTime * timeScale * 1000.0;
  }
  else
  {
    tickCount = timeGetTime();	 
  }

  return tickCount;
}