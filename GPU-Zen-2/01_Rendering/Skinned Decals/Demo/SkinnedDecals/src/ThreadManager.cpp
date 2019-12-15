#include <stdafx.h>
#include <Demo.h>
#include <ThreadManager.h>

Thread::Thread(ThreadFunction threadMain, void *data)
{
	assert((threadMain != nullptr) && (data != nullptr));
	handle = CreateThread(nullptr, 0, (LPTHREAD_START_ROUTINE)threadMain, data, 0, nullptr);  
	assert(handle != nullptr);
}

Thread::~Thread()
{
	CloseHandle(handle);
}

void Thread::Join()
{
	WaitForSingleObject(handle,INFINITE);
}


DWORD ThreadManager::DemoWorkerThread(void *data)
{
	ThreadManager *threadManager = (ThreadManager*)data;
	while(1)
	{
		threadManager->Lock();
		while(threadManager->numTasks == 0)
		{
			threadManager->numWaitingThreads++;
			threadManager->Unlock();
			WaitForSingleObject(threadManager->event, INFINITE);	
			if(threadManager->quit)
			{
				threadManager->Lock();
				threadManager->numWaitingThreads--;
				threadManager->Unlock();
				return 0;
			}
			threadManager->Lock();
			threadManager->numWaitingThreads--;
		}
		ThreadedTask *task = threadManager->tasks[--threadManager->numTasks];
		threadManager->Unlock();
		task->taskState = TASK_IN_PROGRESS;
		task->Run();
		task->taskState = TASK_FINISHED;
	}
	return 0;
}

void ThreadManager::Shutdown()
{
#ifdef MULTITHREADING
	DeleteThreads();
	CloseHandle(event);
	DeleteCriticalSection(&criticalSection);
#endif
}

bool ThreadManager::Init()
{	
#ifdef MULTITHREADING
	InitializeCriticalSection(&criticalSection);
	event = CreateEvent(nullptr, false, false, nullptr);
	if(!event)
		return false;
	tasks.Resize(INIT_TASK_POOL_SIZE);
	if(!CreateThreads())
		return false;
#endif

	return true;
}

bool ThreadManager::CreateThreads(UINT threadCount)
{
	quit = false;
	if(threadCount == 0)
	{
		SYSTEM_INFO systemInfo;
		GetSystemInfo(&systemInfo); 
		numThreads = max(systemInfo.dwNumberOfProcessors - 1, 1);
	}
	else
		numThreads = threadCount;
	threads = new Thread*[numThreads];
	if(!threads)
		return false;
	for(UINT i=0; i<numThreads; i++)
	{
		threads[i] = new Thread(DemoWorkerThread, this);
		if(!threads[i])
			return false;
	}
	return true;
}

void ThreadManager::DeleteThreads()
{
	WaitForTasks();
	quit = true;
	while(numWaitingThreads > 0)
	{
		Lock();
		SetEvent(event);
		Unlock();
	}
	for(UINT i=0; i<numThreads; i++)
		threads[i]->Join();
	for(UINT i=0; i<numThreads; i++)
		SAFE_DELETE(threads[i]);
	SAFE_DELETE_ARRAY(threads);
}

void ThreadManager::ScheduleTask(ThreadedTask *task)
{
  assert(task != nullptr);
#ifdef MULTITHREADING
	task->taskState = TASK_PENDING;
	Lock();
	if(tasks.GetCapacity() < (numTasks + 1))
		tasks.Resize(numTasks + INIT_TASK_POOL_SIZE);
	tasks[numTasks++] = task;
	SetEvent(event);
	Unlock();
#else
	task->taskState = TASK_IN_PROGRESS;
  task->Run();
	task->taskState = TASK_FINISHED;
#endif
}

void ThreadManager::WaitForTasks()
{ 
#ifdef MULTITHREADING
	while(!((numTasks == 0) && (numWaitingThreads == numThreads)))
	{}
#endif
}

void ThreadManager::Lock()
{
#ifdef MULTITHREADING
	EnterCriticalSection(&criticalSection);
#endif
}

void ThreadManager::Unlock()
{
#ifdef MULTITHREADING
	LeaveCriticalSection(&criticalSection);
#endif 
}

bool ThreadManager::SetThreadCount(UINT threadCount)
{
#ifdef MULTITHREADING
	if(threadCount == numThreads)
		return true;
  DeleteThreads();
	if(!CreateThreads(threadCount))
		return false;
#endif

	return true;
}
