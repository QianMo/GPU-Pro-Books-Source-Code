/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beJob.h"

#include "beCore/beTask.h"
#include "beCore/beThreadPool.h"

#include <lean/concurrent/atomic.h>
#include <lean/concurrent/spin_lock.h>
#include <lean/concurrent/shareable_spin_lock.h>
#include <lean/concurrent/event.h>

#include <lean/smart/scope_guard.h>
#include <lean/logging/log.h>

/// Implementation of the job class internals.
class beCore::Job::Impl : private beCore::Task
{
private:
	Job *m_pJob;
	JobType::T m_type;
	
	Impl *m_pParent;
	Impl *m_pNextSibling;
	Impl *m_pFirstChild;
	Impl *m_pLastChild;

	lean::spin_lock<> m_runLock;

	volatile int m_waitingForChildren;
	volatile int m_childrenRunning;
	lean::event m_childrenDone;
	lean::shareable_spin_lock<> m_waitLock;

	ThreadPool *m_pPool;

	/// Adds a child to the number of running children.
	void AddRunningChild();
	/// Removes a child from the number of running children. 
	void RemoveRunningChild();
	/// Gets the number of children currently running.
	int GetRunningChildCount();
	/// Waits for all children currently running to terminate.
	void WaitForChildren();

	/// Adds a child job. This method is thread-safe.
	bool AddChild(Impl *pChild);
	/// Gets and unlinks the current child job range. Returns true if valid, false if empty. This method is thread-safe.
	bool GetAndUnlinkChildren(Impl *&pFirstJobImpl, Impl *&pLastJobImpl);
	/// Runs all child jobs.
	void RunChildren();
	/// A child has terminated.
	void ChildTerminated(Impl *pChild);

	/// Runs the job.
	void Run();
	/// Runs the job.
	void DoRun(ThreadPool *pPool);
	/// The job has terminated. A job has terminated when all its child jobs have terminated.
	void Terminated();

public:
	/// Initializes synchronization primitives.
	Impl(Job *pJob, JobType::T type);

	/// Adds a child job. This method is thread-safe.
	bool AddJob(Job *pJob);

	/// Runs the job.
	void Run(ThreadPool *pPool);

	/// Gets the current thread pool.
	LEAN_INLINE ThreadPool* GetThreadPool() const { return m_pPool; }
	/// Gets the job type.
	LEAN_INLINE JobType::T GetType() const { return m_type; }
};

// Constructs an empty job.
beCore::Job::Job(JobType::T type)
	: m_impl( new Impl(this, type) )
{
}

// Destructor.
beCore::Job::~Job()
{
}

// Runs the job.
void beCore::Job::Run(ThreadPool *pPool)
{
	m_impl->Run(pPool);
}

// Adds a child job. This method is thread-safe.
bool beCore::Job::AddJob(Job *pJob)
{
	return m_impl->AddJob(pJob);
}

// Gets the job type.
beCore::JobType::T beCore::Job::GetType() const
{
	return m_impl->GetType();
}

// Initializes synchronization primitives.
beCore::Job::Impl::Impl(Job *pJob, JobType::T type)
		: m_pJob(pJob),
		m_type(type),
		
		m_pParent(nullptr),
		m_pNextSibling(nullptr),
		m_pFirstChild(nullptr),
		m_pLastChild(nullptr),

		m_childrenRunning(0),
		m_waitingForChildren(0),

		m_pPool(nullptr)
{
}

// Runs the job.
LEAN_INLINE void beCore::Job::Impl::Run(ThreadPool *pPool)
{
	if (!pPool)
	{
		LEAN_LOG_ERROR("Nullptr thread pool passed.");
		LEAN_ASSERT_DEBUG(pPool);
		return;
	}

	// Only ever run once at a time
	if (!m_runLock.try_lock())
	{
		LEAN_LOG_ERROR("Job cannot be run twice at the same time.");
		return;
	}

	Impl *pParent = const_cast<Impl *volatile &>(m_pParent);
	
	// Doing this before locking would break atomicity
	if (pParent)
	{
		// WARNING: Job already locked
		m_runLock.unlock();

		LEAN_LOG_ERROR("Cannot manually run child job.");
		LEAN_ASSERT_DEBUG(!pParent);
		return;
	}
	
	// Registering ourselves as our own child takes care of proper waiting and termination
	AddRunningChild();
	lean::scope_annex runningChildGuard = lean::make_scope_annex(this, &Impl::RemoveRunningChild);

	DoRun(pPool);
}

// Runs the job.
void beCore::Job::Impl::Run()
{
	// Registering ourselves as our own child takes care of proper waiting and termination
	AddRunningChild();
	lean::scope_annex runningChildGuard = lean::make_scope_annex(this, &Impl::RemoveRunningChild);

	// ASSERT: Always calls Terminate() when the last child has terminated

	// ASSERT: Child jobs never run more than once at a times
	bool bNotRunningYet = m_runLock.try_lock();
	LEAN_ASSERT(bNotRunningYet);

	// ASSERT: Only child jobs are run this way
	Impl *pParent = const_cast<Impl *volatile &>(m_pParent);
	LEAN_ASSERT(pParent);
	LEAN_ASSERT(pParent->GetThreadPool());

	DoRun(pParent->GetThreadPool());
}

// Runs the job.
LEAN_INLINE void beCore::Job::Impl::DoRun(ThreadPool *pPool)
{
	lean::atomic_set(m_pPool, pPool);

	// Parent always run BEFORE children (sequentially)
	m_pJob->Run();

	RunChildren();
}

// The job has terminated. A job has terminated when all its child jobs have terminated.
LEAN_INLINE void beCore::Job::Impl::Terminated()
{
	m_runLock.unlock();

	// Propagate termination
	if (m_pParent)
		m_pParent->ChildTerminated(this);
}

// Adds a child job. This method is thread-safe.
LEAN_INLINE bool beCore::Job::Impl::AddJob(Job *pJob)
{
	if (!pJob)
	{
		LEAN_LOG_ERROR("nullptr job passed.");
		return false;
	}

	return AddChild(pJob->m_impl.getptr());
}

// Adds a child job. This method is thread-safe.
LEAN_INLINE bool beCore::Job::Impl::AddChild(Impl *pJobImpl)
{
	// Don't intefere with running jobs
	if (!pJobImpl->m_runLock.try_lock())
	{
		LEAN_LOG_ERROR("Job cannot be added while running.");
		return false;
	}

	{
		// Hold the lock until the parent relation has been set
		lean::scoped_sl_lock runLockGuard(pJobImpl->m_runLock, lean::smart::adopt_lock);

		// Atomic, might erroneously be added to several jobs concurrently otherwise
		if (!lean::atomic_test_and_set(pJobImpl->m_pParent, nullptr, this))
		{
			LEAN_LOG_ERROR("Cannot add job that already is part of a job hierarchy.");
			return false;
		}
	}

	Impl *pPrevJobImpl;

	// Wait until up-to-date
	do
	{
		pPrevJobImpl = const_cast<Impl *volatile&>(m_pLastChild);
	}
	// Last child accessed concurrently
	while (!lean::atomic_test_and_set(m_pLastChild, pPrevJobImpl, pJobImpl));

	// Exclusive access, yet use atomics to enfoce order
	lean::atomic_set(
		(pPrevJobImpl)
			? const_cast<Impl *volatile&>(pPrevJobImpl->m_pNextSibling)
			: const_cast<Impl *volatile&>(m_pFirstChild),
		pJobImpl );

	return true;
}

// Gets and unlinks the current child job range. Returns true if valid, false if empty. This method is thread-safe.
LEAN_INLINE bool beCore::Job::Impl::GetAndUnlinkChildren(Impl *&pFirstJobImpl, Impl *&pLastJobImpl)
{
	// Safe to reset as long as last job remains untouched,
	// as first job won't change once valid
	pFirstJobImpl = lean::atomic_set(m_pFirstChild, nullptr);

	// Nothing to run in an empty job queue
	if (!pFirstJobImpl)
		return false;

	// Wait until up-to-date
	do
	{
		pLastJobImpl = const_cast<Impl *volatile&>(m_pLastChild);
	}
	// Don't lose any jobs while clearing, last child accessed concurrently
	while (!lean::atomic_test_and_set(m_pLastChild, pLastJobImpl, nullptr));

	return true;
}

// Runs all child jobs.
void beCore::Job::Impl::RunChildren()
{
	Impl *pFirstJobImpl, *pLastJobImpl;
	
	// Only process if job range valid
	if (!GetAndUnlinkChildren(pFirstJobImpl, pLastJobImpl))
		return;
	
	if (m_type == JobType::Sequential)
	{
		Impl *pJobImpl, *pNextJobImpl = pFirstJobImpl;

		// Include last job in sequential processing
		do
		{
			pJobImpl = pNextJobImpl;
			LEAN_ASSERT(pJobImpl);

			// Fetch next sibling BEFORE running the current job, the field
			// may become invalid as soon as the current job has terminated
			pNextJobImpl = const_cast<Impl *volatile &>(pJobImpl->m_pNextSibling);

			// Run() always calls RemoveRunningChild() by calling ChildTerminated() via Terminate()
			AddRunningChild();
			pJobImpl->Run();

			// Make sure all parallel child jobs have terminated completely
			if (pJobImpl->m_type != JobType::Sequential)
				pJobImpl->WaitForChildren();
		}
		while (pJobImpl != pLastJobImpl);
	}
	else
	{
		Impl *pJobImpl = pFirstJobImpl;

		// Exclude last job from parallel processing
		while (pJobImpl != pLastJobImpl)
		{
			LEAN_ASSERT(pJobImpl);

			// Fetch next sibling BEFORE running the current job, the field
			// may become invalid as soon as the current job has terminated
			Impl *pNextJobImpl = const_cast<Impl *volatile &>(pJobImpl->m_pNextSibling);

			// Run() always calls RemoveRunningChild() by calling ChildTerminated() via Terminate()
			AddRunningChild();
			m_pPool->AddTask(pJobImpl);

			pJobImpl = pNextJobImpl;
		}

		// 'inline' last child to avoid top-level waiting:
		// -> parent parallel: ALL siblings run & scheduled in different threads anyways
		// -> parent sequential: would have to wait for this child to terminate anyways

		// Run() always calls RemoveRunningChild() by calling ChildTerminated() via Terminate()
		AddRunningChild();
		pJobImpl->Run();
	}
}

// A child has terminated.
LEAN_INLINE void beCore::Job::Impl::ChildTerminated(Impl *pChild)
{
	// Detach resetting parent LAST (parent implicitly locks job)
	lean::atomic_set(pChild->m_pNextSibling, nullptr);
	lean::atomic_set(pChild->m_pParent, nullptr);

	RemoveRunningChild();
}

// Adds a child to the number of running children.
LEAN_INLINE void beCore::Job::Impl::AddRunningChild()
{
	lean::atomic_increment(m_childrenRunning);
}

// Removes a child from the number of running children. 
LEAN_INLINE void beCore::Job::Impl::RemoveRunningChild()
{
	if (lean::atomic_decrement(m_childrenRunning) == 0)
	{
		// ASSERT: No more children added until Terminated() has been called
		
		if (m_waitingForChildren > 0)
		{
			// Prevent waiting threads from being added or removed until
			// all currently waiting threads have been signaled
			lean::scoped_ssl_lock waitLock(m_waitLock);

			// Double check
			// -> Waiting threads might have been added before locking
			// -> Prevent signaling when no threads are waiting (signal would not be reset)
			if (m_waitingForChildren > 0)
				m_childrenDone.set();
		}

		// A job has terminated when all its child jobs have terminated.
		Terminated();
	}
}

// Gets the number of children currently running.
LEAN_INLINE int beCore::Job::Impl::GetRunningChildCount()
{
	return m_childrenRunning;
}

// Waits for all children currently running to terminate.
LEAN_INLINE void beCore::Job::Impl::WaitForChildren()
{
	// Only wait if the job has not yet terminated
	// -> Allows for lazy signaling to minimize syscalls
	// -> Allows lazy signal to recover from ALL waiting threads
	if (m_childrenRunning > 0)
	{
		{
			// Prevent waiting threads from being added or removed until
			// all currently waiting threads have been signaled
			lean::scoped_ssl_lock_shared waitLock(m_waitLock);

			lean::atomic_increment(m_waitingForChildren);

			// Double check
			// -> We might not get signaled when no threads have been waiting before
			if (m_childrenRunning <= 0)
			{
				// Still in lock scope
				// -> Prevent signaling when no threads are waiting (signal would not be reset)
				lean::atomic_decrement(m_waitingForChildren);
				return;
			}
		}

		m_childrenDone.wait();
		
		// Reset signal as soon as all waiting threads have been released
		if (lean::atomic_decrement(m_waitingForChildren) == 0)
			m_childrenDone.reset();
	}
}
