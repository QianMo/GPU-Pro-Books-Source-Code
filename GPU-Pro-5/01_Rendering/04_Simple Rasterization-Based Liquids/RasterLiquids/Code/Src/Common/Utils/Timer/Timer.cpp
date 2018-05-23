
#include <Common\Utils\Timer\Timer.hpp>

#include <windows.h>

///<
class TimerImpl
{
	__int64 m_0, m_1, m_f;

public:

	void	Begin()
	{
		m_0=0;m_1=0;m_f=0;

		int32 pf = QueryPerformanceCounter((LARGE_INTEGER*)&m_0);
		ASSERT(pf!=0, "Error performance querry!  ");
	}

	float32 ElapsedTime()
	{
		int32 pf = QueryPerformanceCounter((LARGE_INTEGER*)&m_1);
		ASSERT(pf!=0, "Error performance querry!  ");

		pf = QueryPerformanceFrequency((LARGE_INTEGER *)&m_f);

		return static_cast<float32>(m_1-m_0)/static_cast<float32>(m_f);
	}
};


Timer::Timer() : m_pImpl(0)
{
	m_pImpl=new TimerImpl();
}

Timer::~Timer()
{
	M::Delete(&m_pImpl);
}

void	Timer::Begin			(){ 	if(m_pImpl) m_pImpl->Begin(); }
float32 Timer::ElapsedTime		(){ 	if(m_pImpl)	return m_pImpl->ElapsedTime(); else return 0; }


