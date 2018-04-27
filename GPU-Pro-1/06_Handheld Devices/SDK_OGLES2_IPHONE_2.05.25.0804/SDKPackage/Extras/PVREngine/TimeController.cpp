/******************************************************************************

 @File         TimeController.cpp

 @Title        TimeController

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Keeps track of time since last frame, direction of animation etc.

******************************************************************************/
#include "TimeController.h"
#include "math.h"

namespace pvrengine
{

	/******************************************************************************/

	TimeController::TimeController()
	{
		m_bFrozen= false;
		m_bForwards = true;
		m_i32DeltaTimeTotal =0;
		m_i32FrameCount = 0;
		m_i32FPS = 0;
		m_fAnimationSpeed = 0.f;
	}

	/******************************************************************************/

	float TimeController::getFrame(const int iTime)
	{
		int i32DeltaTime = iTime - m_iTimePrev;
		m_i32DeltaTimeTotal += i32DeltaTime;
		m_fDeltaTime = ((float)i32DeltaTime)/100.f;
		m_i32FrameCount++;
		if(m_i32DeltaTimeTotal>=1000)		// a second is 1000 milliseconds
		{
			m_i32FPS = m_i32FrameCount;
			m_i32DeltaTimeTotal = 0;
			m_i32FrameCount = 0;
		}
		m_iTimePrev	= iTime;
		if(!(m_bFrozen))
		{
			if(m_bForwards)
				m_fFrame += m_fDeltaTime*(float)pow(2.0f,m_fAnimationSpeed);
			else
				m_fFrame -= m_fDeltaTime*(float)pow(2.0f,m_fAnimationSpeed);
		}

		// ensure looping of animation is successful and smooth
		while (m_fFrame >= (float)(m_i32NumFrames-1))
		{
			m_fFrame -= (float)(m_i32NumFrames-1);
		}
		while(m_fFrame<0.0f)
		{
			m_fFrame += (float)(m_i32NumFrames-1);
		}
		return m_fFrame;
	}

	/******************************************************************************/

	void TimeController::rewind()
	{
		m_fFrame-=8.0f*(float)pow(2.0f,m_fAnimationSpeed);
	}

	/******************************************************************************/

	void TimeController::fastforward()
	{
		m_fFrame+=8.0f*(float)pow(2.0f,m_fAnimationSpeed);
	}

}

/******************************************************************************
End of file (TimeController.cpp)
******************************************************************************/
