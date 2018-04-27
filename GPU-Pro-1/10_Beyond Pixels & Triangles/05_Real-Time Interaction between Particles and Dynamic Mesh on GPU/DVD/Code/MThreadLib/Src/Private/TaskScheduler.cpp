#include "Precompiled.h"

#include "MTTask.h"

#include "TaskSchedulerImpl.h"

#include "TaskSchedulerConfig.h"
#include "TaskScheduler.h"

#define MD_NAMESPACE TaskSchedulerNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	TaskScheduler::TaskScheduler( const TaskSchedulerConfig& cfg ) :
	Parent( cfg )
	{
		mImpl.reset( new TaskSchedulerImpl( cfg ) );
	}

	//------------------------------------------------------------------------

	TaskScheduler::~TaskScheduler() 
	{
	}

	//------------------------------------------------------------------------

	void
	TaskScheduler::ParallelForImpl( const void* data, size_t elem_size, size_t count, MTTask& task )
	{
		using namespace tbb;

		struct TaskWrapper
		{
			void operator()( const blocked_range<size_t>& r ) const
			{
				for( size_t i = r.begin(), e = r.end(); i < e; i++ )
				{
					task->Process(	(char*)data + i * elem_size );
				}
			}

			const void* data;
			size_t elem_size;
			MTTask* task;
		} taskWrapper = { data, elem_size, &task };

		parallel_for( blocked_range<size_t>( 0, count ), taskWrapper );
	}

	//------------------------------------------------------------------------

	void
	TaskScheduler::ParallelForImpl( const void* data, size_t elem_size, size_t count, MTTaskIndexed& task )
	{
		using namespace tbb;

		struct TaskWrapper
		{
			void operator()( const blocked_range<size_t>& r ) const
			{
				for( size_t i = r.begin(), e = r.end(); i < e; i++ )
				{
					task->Process(	(char*)data + i * elem_size, i );
				}
			}

			const void* data;
			size_t elem_size;
			MTTaskIndexed* task;
		} taskWrapper = { data, elem_size, &task };

		parallel_for( blocked_range<size_t>( 0, count ), taskWrapper );
	}
}