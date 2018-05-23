/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beThreadPool.h"

#include <lean/concurrent/thread.h>

#include <lean/concurrent/atomic.h>
#include <lean/concurrent/critical_section.h>
#include <lean/concurrent/event.h>
#include <lean/concurrent/semaphore.h>

#include <lean/functional/callable.h>
#include <lean/smart/scope_guard.h>

#include <deque>

#include <lean/logging/log.h>

/// Implements a thread pool.
class beCore::ThreadPool::Impl
{
private:
	volatile int m_nActiveThreads;
	volatile bool m_bShuttingDown;

	lean::critical_section m_threadsLock;
	lean::event m_threadsTerminated;

	lean::critical_section m_tasksLock;
	std::deque<Task*> m_tasks;

	volatile int m_nIdleCount;
	lean::semaphore m_idleBlock;

	/// Launches a new worker thread. This method is thread-safe.
	void LaunchWorker();
	/// To be called when a worker thread terminates. This method is thread-safe.
	void WorkerTerminated();

	/// Shuts down all worker threads. This method is thread-safe.
	void ShutDownWorkers();

	/// Schedules the next tasks. This method is thread-safe.
	void WorkerThread();

	/// Gets the next task scheduled for execution, nullptr if no such task. This method is thread-safe.
	Task* NextTask();

public:
	/// Constructs the given number of threads.
	Impl(size_t threadCount);
	/// Waits for all active threads to terminate.
	~Impl();

	/// Adds the given task to be executed when a thread becomes available. This method is thread-safe.
	void AddTask(Task *pTask);
};

// Constructs a thread pool maintaining the given number of threads.
beCore::ThreadPool::ThreadPool(size_t threadCount)
	: m_impl(new Impl(threadCount))
{
}

// Destructor.
beCore::ThreadPool::~ThreadPool()
{
}

// Adds the given task to be executed when a thread becomes available. This method is thread-safe.
void beCore::ThreadPool::AddTask(Task *pTask)
{
	m_impl->AddTask(pTask);
}

// Constructs the given number of threads.
beCore::ThreadPool::Impl::Impl(size_t threadCount)
	: m_nActiveThreads(0),
	m_bShuttingDown(false),
	m_threadsTerminated(true),

	m_nIdleCount(0),
	m_idleBlock(0)
{
	try
	{
		for (size_t i = 0; i < threadCount; ++i)
			LaunchWorker();
	}
	catch(...)
	{
		ShutDownWorkers();
		throw;
	}
}

// Waits for all active threads to terminate.
beCore::ThreadPool::Impl::~Impl()
{
	ShutDownWorkers();
}

// Adds the given task to be executed when a thread becomes available. This method is thread-safe.
LEAN_INLINE void beCore::ThreadPool::Impl::AddTask(Task *pTask)
{
	if (!pTask)
	{
		LEAN_LOG_ERROR("nullptr task passed.");
		return;
	}

	{
		// Tasks accessed concurrently
		lean::scoped_cs_lock lock(m_tasksLock);

		m_tasks.push_back(pTask);
	}

	// Wait until up to date
	for (int nIdleCount; (nIdleCount = m_nIdleCount) > 0; )
	{
		// Idle count accessed concurrently
		if (lean::atomic_test_and_set(m_nIdleCount, nIdleCount, nIdleCount - 1))
		{
			m_idleBlock.unlock();
			break;
		}
	}
}

// Gets the next task scheduled for execution, nullptr if no such task. This method is thread-safe.
beCore::Task* beCore::ThreadPool::Impl::NextTask()
{
	Task *pTask = nullptr;
	
	// Tasks accessed concurrently
	lean::scoped_cs_lock lock(m_tasksLock);
	
	if (!m_tasks.empty())
	{
		pTask = m_tasks.front();
		m_tasks.pop_front();
	}

	return pTask;
}

// Schedules the next tasks. This method is thread-safe.
void beCore::ThreadPool::Impl::WorkerThread()
{
	// Properly terminate thread on uncaught exceptions
	lean::scope_annex terminateGuard = lean::make_scope_annex(this, &Impl::WorkerTerminated);

	while (!m_bShuttingDown)
	{
		Task *pTask = NextTask();
		
		if (pTask)
			// Instantly run next task
			pTask->Run();
		else
		{
			// Spin for a short while, waiting for the next task
			for (int i = 0; !pTask && i < 4096; ++i)
				pTask = NextTask();

			if (pTask)
				// If we're lucky, we've got another task now
				pTask->Run();
			else
			{
				// Activate incoming task signaling
				lean::atomic_increment(m_nIdleCount);

				pTask = NextTask();

				// Double check
				// -> A job might have been added before we signaled idle state
				if (pTask)
				{
					// The idling semaphore might have already been released, in which case
					// future idling might incur some prolonged spinning (harmless)
					// -> To prevent this behavior, another lock would be required
					lean::atomic_decrement(m_nIdleCount);

					// If we're lucky, we've got another task now
					pTask->Run();
				}
				else
					// Wait for busier days otherwise, without wasting any further resources
					m_idleBlock.lock();
			}
		}
	}
}

// Shuts down all worker threads. This method is thread-safe.
void beCore::ThreadPool::Impl::ShutDownWorkers()
{
	{
		// Make sure any initiated launches have been fully completed
		lean::scoped_cs_lock lock(m_threadsLock);
		m_bShuttingDown = true;
	}

	// ASSERT: no more threads created from this point on

	// Wake up ALL idling threads
	for (int i = m_nIdleCount + m_nActiveThreads; i-- > 0; )
	{
		lean::atomic_decrement(m_nIdleCount);
		m_idleBlock.unlock();
	}

	m_threadsTerminated.wait();
}

// Launches a new worker thread. This method is thread-safe.
void beCore::ThreadPool::Impl::LaunchWorker()
{
	bool create;

	{
		// Atomically check, increment & signal
		lean::scoped_cs_lock lock(m_threadsLock);

		// Don't add new threads on shut-down
		create = !m_bShuttingDown;

		if (create)
			if (m_nActiveThreads++ == 0)
				m_threadsTerminated.reset();
	}

	if (create)
	{
		try
		{
			lean::thread( lean::make_callable(this, &Impl::WorkerThread) );
		}
		catch(...)
		{
			WorkerTerminated();
			throw;
		}
	}
}

// To be called when a worker thread terminates. This method is thread-safe.
void beCore::ThreadPool::Impl::WorkerTerminated()
{
	// Atomically decrement & signal
	lean::scoped_cs_lock lock(m_threadsLock);

	if (--m_nActiveThreads == 0)
		m_threadsTerminated.set();
}
