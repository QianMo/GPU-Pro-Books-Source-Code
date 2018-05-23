/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_JOB
#define BE_CORE_JOB

#include "beCore.h"
#include <lean/tags/noncopyable.h>
#include <lean/pimpl/pimpl_ptr.h>

namespace beCore
{

// Prototypes
class Task;
class ThreadPool;

/// Job type enumeration.
struct JobType
{
	/// Job type enumeration.
	enum T
	{
		Sequential,	///< Run children sequentially
		Parallel	///< Run children in parallel
	};
	LEAN_MAKE_ENUM_STRUCT(JobType)
};

/// Job class that allows for the distribution of job queues on multiple cores.
class Job : public lean::noncopyable
{
private:
	class Impl;
	lean::pimpl_ptr<Impl> m_impl;

protected:
	/// Performs some actual work. Does not need to be called by overriding classes.
	virtual void Run() { }

public:
	/// Constructs an empty job.
	BE_CORE_API Job(JobType::T type);
	/// Destructor.
	BE_CORE_API ~Job();
	
	/// Runs the job.
	BE_CORE_API void Run(ThreadPool *pPool);

	/// Adds a child job. This method is thread-safe.
	BE_CORE_API bool AddJob(Job *pJob);

	/// Gets the job type.
	BE_CORE_API JobType::T GetType() const;
};

} // namespace

#endif