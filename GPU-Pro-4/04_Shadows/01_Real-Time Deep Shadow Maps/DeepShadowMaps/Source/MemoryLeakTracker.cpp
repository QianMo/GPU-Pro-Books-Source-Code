#include "MemoryLeakTracker.h"

#if defined(_MSC_VER) && defined(WIN32) && defined(_DEBUG)
	_CrtMemState mem_state;
#endif



