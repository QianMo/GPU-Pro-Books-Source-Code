#ifndef __ANIM_TIMER_H__
#define __ANIM_TIMER_H__

#include <time.h>

struct ANIMATION_TIMER {

   enum REQUEST {
      GET_FRAME,
      RESET_FRAME
   };

   static const int FPS = 25.f;

   static int requestFrame(REQUEST request = GET_FRAME)
   {      
      static int     lastFrame   =  0;
      static clock_t lastTick    =  clock();

      if (request == RESET_FRAME) {
         lastTick = clock();
      }

      lastFrame = (int((clock() - lastTick) * FPS / 1000.f));

      return lastFrame;
   }   
};

#endif // __ANIM_TIMER_H__