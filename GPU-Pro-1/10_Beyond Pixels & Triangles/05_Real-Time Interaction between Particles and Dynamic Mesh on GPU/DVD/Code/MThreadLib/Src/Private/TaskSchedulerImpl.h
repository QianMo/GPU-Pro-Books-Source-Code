#ifndef MTHREADLIB_TASKSCHEDULERIMPL_H_INCLUDED
#define MTHREADLIB_TASKSCHEDULERIMPL_H_INCLUDED

#include "Forw.h"

namespace Mod
{
	class TaskSchedulerImpl
	{
		// construction/ destruction
	public:
		TaskSchedulerImpl( const TaskSchedulerConfig& cfg );
		~TaskSchedulerImpl();

	private:
		tbb::task_scheduler_init mTBBInit;
	};
}

#endif