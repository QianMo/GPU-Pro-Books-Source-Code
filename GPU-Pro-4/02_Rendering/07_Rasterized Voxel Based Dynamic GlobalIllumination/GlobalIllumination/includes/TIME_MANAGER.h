#ifndef TIME_MANAGER_H
#define TIME_MANAGER_H

// TIME_MANAGER
//  Simple manager for time operations.
class TIME_MANAGER
{
public:
  TIME_MANAGER()
	{
    performanceCounterAvailable = true;
		timeScale = 0.0;
		frameInterval = 0.0;
		fps = 0.0f;
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

private:
	bool performanceCounterAvailable;
	double timeScale,frameInterval;
	float fps;

};

#endif