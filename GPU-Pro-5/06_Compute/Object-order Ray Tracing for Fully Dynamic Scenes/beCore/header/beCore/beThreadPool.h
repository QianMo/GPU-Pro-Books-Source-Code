/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_THREAD_POOL
#define BE_CORE_THREAD_POOL

#include "beCore.h"
#include "beTask.h"
#include <lean/tags/noncopyable.h>
#include <lean/pimpl/pimpl_ptr.h>

namespace beCore
{

/// Thread pool class that allows for the distribution of separate tasks on multiple cores.
class ThreadPool : public lean::noncopyable
{
private:
	class Impl;
	lean::pimpl_ptr<Impl> m_impl;
	
public:
	/// Constructs a thread pool maintaining the given number of threads.
	BE_CORE_API ThreadPool(size_t threadCount);
	/// Destructor.
	BE_CORE_API ~ThreadPool();
	
	/// Adds the given task to be executed when a thread becomes available. This method is thread-safe.
	BE_CORE_API void AddTask(Task *pTask);
};

} // namespace

#endif