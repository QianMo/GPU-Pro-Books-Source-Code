#ifndef MTHREADLIB_PRECOMPILED_H_INCLUDED
#define MTHREADLIB_PRECOMPILED_H_INCLUDED

#include "PrecompiledCommon.h"

#ifdef MD_WIN_PLATFORM

#pragma warning(push)
#pragma warning( disable: 4512 ) // assigment operator could not be generated
#pragma warning( disable: 4100 ) // unreferenced formal parameter

#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#pragma warning(pop)

#else
#error Unsupported platform!
#endif

#endif