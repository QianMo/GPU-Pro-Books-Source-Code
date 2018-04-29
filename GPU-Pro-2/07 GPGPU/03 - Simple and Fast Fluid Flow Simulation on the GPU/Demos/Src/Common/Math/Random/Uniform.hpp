#ifndef __UNIFORM_HPP__
#define __UNIFORM_HPP__

#include <time.h>
#include <stdlib.h>
#include <Common/System/Types.hpp>
#include <Common/Incopiable.hpp>

class Uniform : public Incopiable
{
	static bool ms_bInitialized;
public:
	static uint32 Rand()
	{ 
		if (ms_bInitialized) 
		{
			return rand(); 
		}
		else
		{
			time(NULL);
			ms_bInitialized=true;
			return Rand();
		}
	}

	static float32 Randf(){return (float32)Rand()/((float32)(RAND_MAX+1));}

};

#endif