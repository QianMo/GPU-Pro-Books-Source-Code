#include "TimerQuery.h"
#include <iostream>

TimerQuery::TimerQuery()
{
   glGenQueries(1, &timerQuery);
   queried = false;
   accumTimeElapsed = 0;
   accumCounter = 0;
}

TimerQuery::~TimerQuery()
{
   glDeleteQueries(1, &timerQuery);
}

void TimerQuery::start(bool doIt)
{
   //return;
   if(!doIt) return;
   available = 0;
   queried = false;
   glBeginQuery(GL_TIME_ELAPSED_EXT, timerQuery);
}


void TimerQuery::end(bool doIt)
{
   //return;
   if(!doIt) return;
   glEndQuery(GL_TIME_ELAPSED_EXT);
   queried = true;
}

bool TimerQuery::waitAndAccumResult()
{
   //return false;
   GLuint64EXT time = waitAndGetResult();
   if(time != 0)
   {
      accumTimeElapsed += time; 
      accumCounter++;
   }
   return (time != 0);
}

double TimerQuery::getAccumAverageResult()
{
   double ms = double(accumTimeElapsed)/1000000.0/accumCounter;
   accumTimeElapsed = 0;
   accumCounter = 0;
   return ms;
}

GLuint64EXT TimerQuery::waitAndGetResult()
{
   //return 0;
   if(queried)
   {
      queried = false;
      //int counter = 0;
      while (!available)
      {
         //if(counter > 10000) return 0; // do not wait any more
         //std::cout << "not available " << counter << std::endl;
         glGetQueryObjectiv(timerQuery, GL_QUERY_RESULT_AVAILABLE, &available);
         //counter++;
      }
      glGetQueryObjectui64vEXT(timerQuery, GL_QUERY_RESULT, &timeElapsed);
      return timeElapsed;
   }
   return 0;
}
