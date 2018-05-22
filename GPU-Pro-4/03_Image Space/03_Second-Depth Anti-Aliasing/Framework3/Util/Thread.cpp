
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "Thread.h"

#ifdef _WIN32

ThreadHandle createThread(ThreadProc startProc, void *param){
	return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) startProc, param, 0, NULL);
}

void deleteThread(ThreadHandle thread){
	CloseHandle(thread);
}

void waitOnThread(const ThreadHandle threadID){
	WaitForSingleObject(threadID, INFINITE);
}

void waitOnAllThreads(const ThreadHandle *threads, const int nThreads){
	WaitForMultipleObjects(nThreads, threads, TRUE, INFINITE);
}

void waitOnAnyThread(const ThreadHandle *threads, const int nThreads){
	WaitForMultipleObjects(nThreads, threads, FALSE, INFINITE);
}

void createMutex(Mutex &mutex){
	mutex = CreateMutex(NULL, FALSE, NULL);
}

void deleteMutex(Mutex &mutex){
	CloseHandle(mutex);
}

void lockMutex(Mutex &mutex){
	WaitForSingleObject(mutex, INFINITE);
}

void unlockMutex(Mutex &mutex){
	ReleaseMutex(mutex);
}

void createCondition(Condition &condition){
	condition.waiters_count = 0;
	condition.was_broadcast = false;
	condition.sema = CreateSemaphore(NULL, 0, 0x7fffffff, NULL);
	InitializeCriticalSection(&condition.waiters_count_lock);
	condition.waiters_done = CreateEvent(NULL, FALSE, FALSE, NULL);
}

void deleteCondition(Condition &condition){
	CloseHandle(condition.sema);
	DeleteCriticalSection(&condition.waiters_count_lock);
	CloseHandle(condition.waiters_done);
}

void waitCondition(Condition &condition, Mutex &mutex){
	EnterCriticalSection(&condition.waiters_count_lock);
	condition.waiters_count++;
	LeaveCriticalSection(&condition.waiters_count_lock);

	SignalObjectAndWait(mutex, condition.sema, INFINITE, FALSE);

	EnterCriticalSection(&condition.waiters_count_lock);
		// We're no longer waiting...
		condition.waiters_count--;
		// Check to see if we're the last waiter after broadcast.
		bool last_waiter = condition.was_broadcast && (condition.waiters_count == 0);
	LeaveCriticalSection(&condition.waiters_count_lock);

	// If we're the last waiter thread during this particular broadcast then let all the other threads proceed.
	if (last_waiter){
		// This call atomically signals the <waiters_done> event and waits until
		// it can acquire the <mutex>. This is required to ensure fairness.
		SignalObjectAndWait(condition.waiters_done, mutex, INFINITE, FALSE);
	} else {
		// Always regain the external mutex since that's the guarantee we give to our callers.
		WaitForSingleObject(mutex, INFINITE);
	}
}

void signalCondition(Condition &condition){
	EnterCriticalSection(&condition.waiters_count_lock);
		bool have_waiters = (condition.waiters_count > 0);
	LeaveCriticalSection(&condition.waiters_count_lock);

	// If there aren't any waiters, then this is a no-op.
	if (have_waiters){
		ReleaseSemaphore(condition.sema, 1, 0);
	}
}


void broadcastCondition(Condition &condition){
	// This is needed to ensure that <waiters_count> and <was_broadcast> are consistent relative to each other.
	EnterCriticalSection(&condition.waiters_count_lock);
	bool have_waiters = false;

	if (condition.waiters_count > 0){
		// We are broadcasting, even if there is just one waiter...
		// Record that we are broadcasting, which helps optimize
		// <pthread_cond_wait> for the non-broadcast case.
		condition.was_broadcast = true;
		have_waiters = true;
	}

	if (have_waiters){
		// Wake up all the waiters atomically.
		ReleaseSemaphore(condition.sema, condition.waiters_count, 0);

		LeaveCriticalSection(&condition.waiters_count_lock);

		// Wait for all the awakened threads to acquire the counting semaphore.
		WaitForSingleObject(condition.waiters_done, INFINITE);
		// This assignment is okay, even without the <waiters_count_lock> held
		// because no other waiter threads can wake up to access it.
		condition.was_broadcast = false;
	} else {
		LeaveCriticalSection(&condition.waiters_count_lock);
	}
}




#else

ThreadHandle createThread(ThreadProc startProc, void *param){
	pthread_t th;
	pthread_create(&th, NULL, (void *(*)(void *)) startProc, param);

	return th;
}

void deleteThread(ThreadHandle thread){

}

void waitOnThread(const ThreadHandle threadID){
	pthread_join(threadID, NULL);
}

void createMutex(Mutex &mutex){
	pthread_mutex_init(&mutex, NULL);
}

void deleteMutex(Mutex &mutex){
	pthread_mutex_destroy(&mutex);
}

void lockMutex(Mutex &mutex){
	pthread_mutex_lock(&mutex);
}

void unlockMutex(Mutex &mutex){
	pthread_mutex_unlock(&mutex);
}

void createCondition(Condition &condition){
	pthread_cond_init(&condition, NULL);
}

void deleteCondition(Condition &condition){
	pthread_cond_destroy(&condition);
}

void waitCondition(Condition &condition, Mutex &mutex){
	pthread_cond_wait(&condition, &mutex);
}

void signalCondition(Condition &condition){
	pthread_cond_signal(&condition);
}

void broadcastCondition(Condition &condition){
	pthread_cond_broadcast(&condition);
}

#endif


//#include <stdio.h>

struct ThreadParam {
	Thread *thread;
	int threadInstance;
};

#ifdef _WIN32

DWORD WINAPI threadStarter(void *param){
//	((Thread *) startFunc)->mainFunc(0);

	Thread *thread = ((ThreadParam *) param)->thread;
	int instance = ((ThreadParam *) param)->threadInstance;

	delete param;

	thread->mainFunc(instance);

	return 0;
}

void Thread::startThreads(const int threadCount){
	nThreads = threadCount;

	threadHandles = new HANDLE[threadCount];
	threadIDs = new DWORD[threadCount];

	for (int i = 0; i < threadCount; i++){
		ThreadParam *param = new ThreadParam;
		param->thread = this;
		param->threadInstance = i;

		threadHandles[i] = CreateThread(NULL, 0, threadStarter, param, 0, &threadIDs[i]);
	}
}

void Thread::postMessage(const int thread, const int message, void *data, const int size){
	int msg = WM_USER + message;

	int start, end;
	if (thread < 0){
		start = 0;
		end = nThreads;
	} else {
		start = thread;
		end = start + 1;
	}

	for (int i = start; i < end; i++){
		char *msgData = new char[size];
		memcpy(msgData, data, size);

		while (!PostThreadMessage(threadIDs[i], msg, size, (LPARAM) msgData)){
			//printf("PostThreadMessage failed\n");
			Sleep(1);
		}
	}
}

void Thread::mainFunc(const int thread){
	MSG msg;
	while (GetMessage(&msg, NULL, 0, 0) > 0){
		processMessage(thread, msg.message - WM_USER, (void *) msg.lParam, (const int) msg.wParam);
	}
}

void Thread::waitForExit(){
	delete threadIDs;

	//	WaitForSingleObject(threadHandle, INFINITE);
	WaitForMultipleObjects(nThreads, threadHandles, TRUE, INFINITE);

	delete threadHandles;
}

#else

#include <string.h>

void *threadStarter(void *param){
	ThreadParam *tp = (ThreadParam *) param;

	Thread *thread = tp->thread;
	int instance = tp->threadInstance;

	delete tp;

	thread->mainFunc(instance);

	return NULL;
}

void Thread::startThreads(const int threadCount){
/*
	first = last = NULL;

	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&pending, NULL);

	pthread_create(&thread, NULL, threadStarter, this);
*/


	nThreads = threadCount;

	queues = new MessageQueue[threadCount];
	threadHandles = new pthread_t[threadCount];

	for (int i = 0; i < threadCount; i++){
		queues[i].first = NULL;
		queues[i].last  = NULL;
		pthread_mutex_init(&queues[i].mutex, NULL);
		pthread_cond_init(&queues[i].pending, NULL);

		ThreadParam *param = new ThreadParam;
		param->thread = this;
		param->threadInstance = i;

		pthread_create(&threadHandles[i], NULL, threadStarter, param);
	}
}

void Thread::postMessage(const int thread, const int message, void *data, const int size){
	int start, end;
	if (thread < 0){
		start = 0;
		end = nThreads;
	} else {
		start = thread;
		end = start + 1;
	}

	for (int i = start; i < end; i++){
		MessageQueue *queue = queues + i;

		pthread_mutex_lock(&queue->mutex);
			Message *msg = new Message;
			msg->message = message;
			if (data){
				msg->data = new char[size];
				memcpy(msg->data, data, size);
			} else {
				msg->data = NULL;
			}
			msg->size = size;
			msg->next = NULL;

			if (queue->first == NULL){
				queue->first = queue->last = msg;
			} else {
				queue->last->next = msg;
				queue->last = msg;
			}

		pthread_mutex_unlock(&queue->mutex);
		pthread_cond_signal(&queue->pending);
	}
}

void Thread::mainFunc(const int thread){
	bool done = false;
	do {
		Message *msg = getMessage(thread);
		if (msg->message < 0){
			done = true;
		} else {
			processMessage(thread, msg->message, msg->data, msg->size);
		}

		delete msg->data;
		delete msg;
	} while (!done);

	pthread_mutex_destroy(&queues[thread].mutex);
	pthread_cond_destroy(&queues[thread].pending);

//	printf("Done\n");
}

void Thread::waitForExit(){
	for (int i = 0; i < nThreads; i++){
		pthread_join(threadHandles[i], NULL);
	}
}

Message *Thread::getMessage(const int thread){
	MessageQueue *queue = queues + thread;

	pthread_mutex_lock(&queue->mutex);
		while (queue->first == NULL){
			pthread_cond_wait(&queue->pending, &queue->mutex);
		}
		Message *ret = queue->first;
		queue->first = queue->first->next;
		if (queue->first == NULL) queue->last = NULL;
	pthread_mutex_unlock(&queue->mutex);

	return ret;
}

#endif // !_WIN32
