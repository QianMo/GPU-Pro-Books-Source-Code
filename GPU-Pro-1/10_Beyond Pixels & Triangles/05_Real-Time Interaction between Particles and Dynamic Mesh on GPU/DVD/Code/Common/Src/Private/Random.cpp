#include "Precompiled.h"

#include "Random.h"

namespace Mod
{
	float URnd()
	{
		return float( rand() ) / RAND_MAX;
	}
}