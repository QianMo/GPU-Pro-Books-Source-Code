#include "Timer.h"

namespace NGraphics
{
    CTimer::CTimer() :
        m_DeltaTime( 0.0f ),
        m_FramesPerSecond( 0.0f )
    {
        __int64 counts_per_second;
        QueryPerformanceFrequency( ( LARGE_INTEGER* )&counts_per_second );
        m_SecondsPerCount = 1.0f / static_cast< float >( counts_per_second );

        QueryPerformanceCounter( ( LARGE_INTEGER* )&m_PreviousTimestamp );
        Tick();
    }

    void CTimer::Tick()
    {
        __int64 timestamp;
        QueryPerformanceCounter( ( LARGE_INTEGER* )&timestamp );
        m_DeltaTime = static_cast< float >( timestamp - m_PreviousTimestamp ) * m_SecondsPerCount;
        m_FramesPerSecond = 1.0f / m_DeltaTime;
        m_PreviousTimestamp = timestamp;
    }

    const float CTimer::GetDeltaTime() const
    {
        return m_DeltaTime;
    }
    const float CTimer::GetFramesPerSecond() const
    {
        return m_FramesPerSecond;
    }
}