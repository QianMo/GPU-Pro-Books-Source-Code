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

#ifdef WIN32
	#include <windows.h>
#endif

#include "SceneGraph.h"

Timer3D::Timer3D()
{
	duration = 1.0f;
	delay = 0.0f;
	cur_time = 0.0f;
	start_time = 0.0f;
	val = 0.0f;
	val_min = 0.0f;
	val_max = 1.0f;
	stopped = true;
	paused = false;
	repeats = 1;
	looping = false;
}

void Timer3D::setDuration(double t)
{
	if (t>0)
		duration = t;
}

void Timer3D::setDelay(double t)
{
	if (t>0)
		delay = t;
}

void Timer3D::setValueRange(double vmin, double vmax)
{
	val_min = vmin;
	val_max = vmax;
	if (!stopped && !paused)
	{
		update();
	}
}

void Timer3D::setRepeats(int rp)
{
	if (rp>=0)
		repeats = rp;
	if (repeats == 0)
		looping = true;
}

void Timer3D::setLooping(bool b)
{
	looping = b;
	if (looping)
		repeats = 0;
}

void Timer3D::start()
{
	if (!paused)
	{
		start_time = GET_TIME();
	}
	else
	{
		double ct = GET_TIME();
		double dt = ct - cur_time;
		start_time = dt;
	}
	paused = false;
	stopped = false;
	update();
}

void Timer3D::stop()
{
	if (!paused && !stopped)
	{	
		update();
	}
	stopped = true;
}

void Timer3D::pause()
{
	if (!paused && !stopped)
	{	
		update();
		paused = true;
	}
}

double Timer3D::getTime()
{
	return cur_time;
}

double Timer3D::getCycleTime()
{
	return cur_time - delay - duration*floor((cur_time - delay)/duration);
}

double Timer3D::getValue()
{
	return val;
}

bool Timer3D::isLooping()
{
	return looping;
}

int Timer3D::getRepeatsTotal()
{
	return repeats;
}

double Timer3D::getRepeatsRemaining()
{
	double rp = (repeats*duration - cur_time+delay)/duration;
	if (rp<0)
		rp = 0.0f;
	return rp;
}

bool Timer3D::terminated()
{
	if (looping)
		return false;
	else
	{
		if (repeats*duration+delay - cur_time >0)
			return false;
		else
			return true;
	}
}

void Timer3D::update()
{
	double s;
	
	if (!paused && !stopped)
	{
		cur_time = GET_TIME() - start_time;
		s = getCycleTime();
		val = ( val_min*(duration-s) + val_max*s )/duration;	
	}
	if (terminated())
		stopped = true;
}

