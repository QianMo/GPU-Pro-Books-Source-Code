#ifndef FPSCOUNTER_H
#define FPSCOUNTER_H

#include "Qt/QSfmlWidget.h"

class FPSCounter
{
public:
   FPSCounter(QSfmlWidget* window);

   float getFPS();


private:
   FPSCounter();
   QSfmlWidget* window;

   int mFramesCounted;
   float mTimeAccumulator;
   float mCurrentFPS;
};

#endif
