#ifndef TIMERQUERY_H
#define TIMERQUERY_H

#include "OpenGL.h"

/// Class for measuring time with the OpenGL extension GL_EXT_timer_query.

class TimerQuery
{
public:
   TimerQuery();
   ~TimerQuery();

   void start(bool doIt = false);
   void end(bool doIt = false);

   GLuint64EXT waitAndGetResult();
   bool waitAndAccumResult();

   /// return average time in ms
   double getAccumAverageResult();

private:

   GLuint timerQuery; ///< holds the ID of our query
   GLuint64EXT timeElapsed; ///< in nano seconds
   GLuint64EXT accumTimeElapsed; // in nano seconds
   GLint available;

   unsigned int accumCounter;

   bool queried;
};

#endif
