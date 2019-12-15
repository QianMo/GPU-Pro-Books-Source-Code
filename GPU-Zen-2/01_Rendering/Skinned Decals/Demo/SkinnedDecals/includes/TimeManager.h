#ifndef TIME_MANAGER_H
#define TIME_MANAGER_H

// TimeManager
//
// Simple manager for time operations.
class TimeManager
{
public:
  TimeManager():
    timeScale(0.0),
    frameInterval(0.0),
    fps(0.0f),
    performanceCounterAvailable(false)
  {
  }

  // checks, whether performance counter is available
  void Init();

  // updates frame-interval/ fps per frame
  void Update(); 

  // gets interval of one frame in ms
  double GetFrameInterval() const
  {
    return frameInterval;
  }

  // gets frames per second
  float GetFPS() const
  {
    return fps;
  }

  // gets current tick count in ms
  double GetTickCount() const;

private:
  double timeScale, frameInterval;
  float fps;
  bool performanceCounterAvailable;

};

#endif