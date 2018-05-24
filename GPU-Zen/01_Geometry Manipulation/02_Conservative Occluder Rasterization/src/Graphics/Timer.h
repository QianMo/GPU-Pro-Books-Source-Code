#pragma once

#include <Windows.h>

namespace NGraphics
{
    class CTimer
    {
    private:
        __int64 m_PreviousTimestamp;
        float m_SecondsPerCount;

        float m_DeltaTime;
        float m_FramesPerSecond;

    public:
        CTimer();

        void Tick();

        const float GetDeltaTime() const;
        const float GetFramesPerSecond() const;
    };
}