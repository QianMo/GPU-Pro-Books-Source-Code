#include <stdafx.h>
#include <Demo.h>
#include <Timer.h>

bool Timer::Update()
{
	elapsedTime += Demo::timeManager->GetFrameInterval();
	timeFraction = elapsedTime / interval;
	if(elapsedTime >= interval)
	{
		elapsedTime = 0.0;
		return true;
	}
	return false;
} 