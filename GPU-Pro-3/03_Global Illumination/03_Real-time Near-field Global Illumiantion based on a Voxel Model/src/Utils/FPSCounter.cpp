#include "FPSCounter.h"

FPSCounter::FPSCounter(QSfmlWidget* window) : window(window)
{
   mCurrentFPS = 0.0f;
   mTimeAccumulator = 0.0f;
   mFramesCounted = 0;
}

float FPSCounter::getFPS()
{
   mTimeAccumulator += window->GetFrameTime(); // in s
   mFramesCounted++;
   if(mTimeAccumulator > 1.0f)
   {
      mCurrentFPS = mFramesCounted / mTimeAccumulator;
      mFramesCounted = 0;
      mTimeAccumulator = 0.0f;
   }
   return mCurrentFPS;

}