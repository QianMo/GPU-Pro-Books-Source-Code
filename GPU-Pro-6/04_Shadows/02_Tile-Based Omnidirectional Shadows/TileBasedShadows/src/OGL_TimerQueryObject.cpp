#include <stdafx.h>
#include <Demo.h>
#include <OGL_TimerQueryObject.h>

void OGL_TimerQueryObject::Release()
{
  if(timerQueryName > 0)
    glDeleteQueries(1, &timerQueryName);
}

bool OGL_TimerQueryObject::Create()
{
  glGenQueries(1, &timerQueryName);
  return true;
}

void OGL_TimerQueryObject::BeginQuery()
{
  if(!queryIssued)
    glBeginQuery(GL_TIME_ELAPSED, timerQueryName);
  cpuStartTickCount = Demo::timeManager->GetTickCount();
}

void OGL_TimerQueryObject::EndQuery()
{
  cpuElapsedTime = Demo::timeManager->GetTickCount()-cpuStartTickCount;
  if(!queryIssued)
  {
    glEndQuery(GL_TIME_ELAPSED);
    queryIssued = true;
  }
}

void OGL_TimerQueryObject::QueryResult()
{
  if(!queryIssued)
    return;

  // check availability of timer query result
  GLuint timerAvailable = GL_FALSE;
  glGetQueryObjectuiv(timerQueryName, GL_QUERY_RESULT_AVAILABLE, &timerAvailable);

  if(timerAvailable == GL_TRUE)
  {
    GLuint timerResult = 0;
    glGetQueryObjectuiv(timerQueryName, GL_QUERY_RESULT, &timerResult);
    gpuElapsedTime = double(timerResult)/1e06;
    queryIssued = false;
  }
}
