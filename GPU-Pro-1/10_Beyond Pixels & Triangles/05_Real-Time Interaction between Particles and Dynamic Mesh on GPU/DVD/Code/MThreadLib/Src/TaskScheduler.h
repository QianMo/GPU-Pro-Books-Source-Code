#ifndef MTHREADLIB_TASKSCHEDULER_H_INCLUDED
#define MTHREADLIB_TASKSCHEDULER_H_INCLUDED

#include "Forw.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE TaskSchedulerNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class TaskScheduler : public TaskSchedulerNS::ConfigurableImpl<TaskSchedulerConfig>
	{
		// types
	public:

		// constructors / destructors
	public:
		explicit TaskScheduler( const TaskSchedulerConfig& cfg );
		~TaskScheduler();
	
		// manipulation/ access
	public:
		template <typename C>
		void ParallelFor( const C& container, MTTask& task );

		template <typename C>
		void ParallelFor( const C& container, MTTaskIndexed& task );

		// helpers
	private:
		void ParallelForImpl( const void* data, size_t elem_size, size_t count, MTTask& task );
		void ParallelForImpl( const void* data, size_t elem_size, size_t count, MTTaskIndexed& task );

		// data
	private:
		TaskSchedulerImplPtr mImpl;

	};

	//------------------------------------------------------------------------

	template <typename C>
	void
	TaskScheduler::ParallelFor( const C& container, MTTask& task )
	{
		MD_STATIC_ASSERT( (is_same< C, Types< C::value_type > :: Vec > :: value) );

		if( !container.empty() )
		{
			ParallelForImpl( &container[0], sizeof C::value_type, container.size(), task );
		}
	}

	//------------------------------------------------------------------------

	template <typename C>
	void
	TaskScheduler::ParallelFor( const C& container, MTTaskIndexed& task )
	{
		MD_STATIC_ASSERT( (is_same< C, Types< C::value_type > :: Vec > :: value) );

		if( !container.empty() )
		{
			ParallelForImpl( &container[0], sizeof C::value_type, container.size(), task );
		}
	}

	

}

#endif