

#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <Common/Common.hpp>

class TimerImpl;

class Timer
{
	
	TimerImpl* m_pImpl;

public:

	Timer();
	~Timer();

	void	Begin		();
	float32 ElapsedTime	();
};


#endif