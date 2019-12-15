#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H

#include <List.h>
#include <ThreadedTask.h>

#define MULTITHREADING 
#define INIT_TASK_POOL_SIZE 128

typedef DWORD (*ThreadFunction)(void*);

// Thread
//
class Thread
{
public:
	Thread(ThreadFunction threadMain, void *data);

	~Thread();

	void Join();

private:
	void *handle;

};


// ThreadManager
//
class ThreadManager
{
public:
	ThreadManager():
	  event(nullptr),
	  threads(nullptr),
		numThreads(0),
		numTasks(0),
		numWaitingThreads(0),
		quit(false)
	{
	}

	~ThreadManager()
	{
		Shutdown();
	}
	
	void Shutdown();

	bool Init();
	
	void ScheduleTask(ThreadedTask *task);

	void WaitForTasks();

	void Lock();

	void Unlock();

	bool SetThreadCount(UINT threadCount=0);

	UINT GetThreadCount() const 
	{
		return numThreads;
	}

private:
	static DWORD DemoWorkerThread(void *data);	

	bool CreateThreads(UINT threadCount=0);

	void DeleteThreads();
	
	CRITICAL_SECTION criticalSection;
	HANDLE event;	
	Thread **threads;
	UINT numThreads;
	List<ThreadedTask*> tasks;
	volatile UINT numTasks;
	volatile UINT numWaitingThreads;
	volatile bool quit;

};

#endif