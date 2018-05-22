//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  Scene Graph 3D                                                          //
//  Georgios Papaioannou, 2009                                              //
//                                                                          //
//  This is a free, extensible scene graph management library that works    //
//  along with the EaZD deferred renderer. Both libraries and their source  //
//  code are free. If you use this code as is or any part of it in any kind //
//  of project or product, please acknowledge the source and its author.    //
//                                                                          //
//  For manuals, help and instructions, please visit:                       //
//  http://graphics.cs.aueb.gr/graphics/                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#ifndef _TIMER3D_
#define _TIMER3D_

#include "time.h"

class Timer3D
{
public:
	Timer3D();
	void setDuration(double t);
	void setDelay(double t);
	void setValueRange(double vmin, double vmax);
	void setRepeats(int rp);
	void setLooping(bool b);
	void start();
	void stop();
	void pause();
	double getTime();
	double getCycleTime();
	double getValue();
	bool isLooping();
	bool isRunning() {return (!stopped)&&(!paused);}
	int getRepeatsTotal();
	double getRepeatsRemaining();
	bool terminated();
	void update();
protected:
	double duration, delay;
	double cur_time, start_time;
	double val, val_min, val_max;
	bool stopped, paused;
	int repeats;
	bool looping;
};

#endif

