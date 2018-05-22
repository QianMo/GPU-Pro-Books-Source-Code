
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

#ifndef _THREAD_H_
#define _THREAD_H_

#include "../Platform.h"

typedef void (*ThreadProc)(void *);

#ifdef _WIN32

typedef HANDLE ThreadHandle;
typedef HANDLE Mutex;

/*
	The condition variable implementation is based on this article:
	http://www.cs.wustl.edu/~schmidt/win32-cv-1.html
*/
typedef struct {
	int waiters_count;
	CRITICAL_SECTION waiters_count_lock;
	HANDLE sema;
	HANDLE waiters_done;
	bool was_broadcast;
} Condition;

#else

#include <pthread.h>

typedef pthread_t ThreadHandle;
typedef pthread_mutex_t Mutex;
typedef pthread_cond_t Condition;

#endif



ThreadHandle createThread(ThreadProc startProc, void *param);
void deleteThread(ThreadHandle thread);
void waitOnThread(const ThreadHandle threadID);
void waitOnAllThreads(const ThreadHandle *threads, const int nThreads);
void waitOnAnyThread(const ThreadHandle *threads, const int nThreads);

void createMutex(Mutex &mutex);
void deleteMutex(Mutex &mutex);
void lockMutex(Mutex &mutex);
void unlockMutex(Mutex &mutex);

void createCondition(Condition &condition);
void deleteCondition(Condition &condition);
void waitCondition(Condition &condition, Mutex &mutex);
void signalCondition(Condition &condition);
void broadcastCondition(Condition &condition);


#define ALL_THREADS (-1)

#ifdef _WIN32

#define THREAD_QUIT (WM_QUIT - WM_USER)

#else

#define THREAD_QUIT (-1)

struct Message {
	int message;
	char *data;
	int size;

	Message *next;
};

struct MessageQueue {
	Message *first, *last;

	pthread_mutex_t mutex;
	pthread_cond_t pending;
};

#endif

class Thread {
public:
	Thread(){}
	virtual ~Thread(){}

	void startThreads(const int threadCount);
	void postMessage(const int thread, const int message, void *data = NULL, const int size = 0);
	void waitForExit();

protected:
	virtual void processMessage(const int thread, const int message, void *data, const int size) = 0;
	void mainFunc(const int thread);

	ThreadHandle *threadHandles;
	int nThreads;

#ifdef _WIN32
	friend DWORD WINAPI threadStarter(void *param);
	DWORD *threadIDs;

#else
	friend void *threadStarter(void *param);

	Message *getMessage(const int thread);

	MessageQueue *queues;

#endif
};

#endif // _THREAD_H_
