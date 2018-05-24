#ifndef OGL_TIMER_QUERY_OBJECT_H
#define OGL_TIMER_QUERY_OBJECT_H

// OGL_TimerQueryObject
//
class OGL_TimerQueryObject
{
public:
  OGL_TimerQueryObject():
    timerQueryName(0),
    cpuStartTickCount(0.0),
    cpuElapsedTime(0.0),
    gpuElapsedTime(0.0),
    queryIssued(false)
  {
  }

  ~OGL_TimerQueryObject()
  {
    Release();
  }

  void Release();

  bool Create();
      
  void BeginQuery();

  void EndQuery();

  void QueryResult();

  double GetCpuElapsedTime() const 
  {
    return cpuElapsedTime;
  }

  double GetGpuElapsedTime() const 
  {
    return gpuElapsedTime;
  }

private: 
  GLuint timerQueryName;
  double cpuStartTickCount;
  double cpuElapsedTime;
  double gpuElapsedTime;
  bool queryIssued;

};

#endif