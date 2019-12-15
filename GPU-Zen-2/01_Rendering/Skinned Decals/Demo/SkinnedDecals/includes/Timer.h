#ifndef TIMER_H
#define TIMER_H

// Timer
//
class Timer
{
public:
  Timer(): interval(0.0)
	{
		Reset();
	}

	void Reset()
	{
		elapsedTime = 0.0;
		timeFraction = 0.0;
	}

	bool Update();     

  void SetInterval(double interval)
  {
    this->interval = interval;
  }

	double GetElapsedTime() const
	{
		return elapsedTime;
	}

	double GetTimeFraction() const
	{
		return timeFraction;
	}

private:
	double elapsedTime;
  double timeFraction;
  double interval;

};

#endif
