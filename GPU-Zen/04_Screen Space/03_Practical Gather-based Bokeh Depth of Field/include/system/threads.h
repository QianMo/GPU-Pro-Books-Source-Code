#pragma once


#include <essentials/assert.h>
#include <essentials/types.h>

#ifdef MAXEST_FRAMEWORK_WINDOWS
	#include <Windows.h>
#else
	#include <pthread.h>
	#include <semaphore.h>
#endif


using namespace NEssentials;


namespace NSystem
{
	#ifdef MAXEST_FRAMEWORK_WINDOWS
		typedef HANDLE ThreadHandle;
		typedef DWORD ThreadEntryReturnValue;
		typedef ThreadEntryReturnValue (__stdcall *ThreadEntry)(void* data);

		typedef HANDLE MutexHandle;

		typedef HANDLE SemaphoreHandle;

		#define THREAD_ENTRY_RETURN_VALUE ThreadEntryReturnValue _stdcall
	#else
		typedef pthread_t* ThreadHandle;
		typedef void* ThreadEntryReturnValue;
		typedef ThreadEntryReturnValue (*ThreadEntry)(void* data);

		typedef pthread_mutex_t* MutexHandle;

		typedef sem_t* SemaphoreHandle;

		#define THREAD_ENTRY_RETURN_VALUE ThreadEntryReturnValue
	#endif

	//

	ThreadHandle ThreadCreate(ThreadEntry threadEntry, void* data = nullptr);
	void ThreadDestroy(ThreadHandle threadHandle);

	MutexHandle MutexCreate();
	void MutexDestroy(MutexHandle mutexHandle);
	bool MutexLock(MutexHandle mutexHandle);
	bool MutexTryLock(MutexHandle mutexHandle);
	void MutexUnlock(MutexHandle mutexHandle);

	SemaphoreHandle SemaphoreCreate(int initialCount, int maxCount);
	void SemaphoreDestroy(SemaphoreHandle semaphoreHandle);
	bool SemaphoreAcquire(SemaphoreHandle semaphoreHandle);
	bool SemaphoreTryAcquire(SemaphoreHandle semaphoreHandle);
	void SemaphoreRelease(SemaphoreHandle semaphoreHandle, int count);

	int32 AtomicIncrement32(int32* data);
	int32 AtomicDecrement32(int32* data);
	int32 AtomicAdd32(int32* data, int x);
	int32 AtomicCompareExchange32(int32* data, int32 comparand, int32 exchange);

	//

	inline ThreadHandle ThreadCreate(ThreadEntry threadEntry, void* data)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			return ::CreateThread(0, 0, threadEntry, data, 0, 0);
		#else
			ThreadHandle threadHandle = new pthread_t();
			pthread_create(threadHandle, nullptr, threadEntry, data);
			return threadHandle;
		#endif
	}

	inline void ThreadDestroy(ThreadHandle threadHandle)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			CloseHandle(threadHandle);
		#else
			delete threadHandle;
		#endif
	}

	//

	inline MutexHandle MutexCreate()
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			return ::CreateMutex(nullptr, false, nullptr);
		#else
			MutexHandle mutexHandle = new pthread_mutex_t();
			pthread_mutex_init(mutexHandle, nullptr);
			return mutexHandle;
		#endif
	}

	inline void MutexDestroy(MutexHandle mutexHandle)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			::CloseHandle(mutexHandle);
		#else
			pthread_mutex_destroy(mutexHandle);
			delete mutexHandle;
		#endif
	}

	inline bool MutexLock(MutexHandle mutexHandle)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			int state = WaitForSingleObject(mutexHandle, INFINITE);
			ASSERT_RELEASE(state == WAIT_OBJECT_0 || state == WAIT_TIMEOUT);
			return state == WAIT_OBJECT_0;
		#else
			return pthread_mutex_lock(mutexHandle) == 0;
		#endif
	}

	inline bool MutexTryLock(MutexHandle mutexHandle)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			int state = WaitForSingleObject(mutexHandle, 0);
			ASSERT_RELEASE(state == WAIT_OBJECT_0 || state == WAIT_TIMEOUT);
			return state == WAIT_OBJECT_0;
		#else
			return pthread_mutex_trylock(mutexHandle) == 0;
		#endif
	}

	inline void MutexUnlock(MutexHandle mutexHandle)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			ReleaseMutex(mutexHandle);
		#else
			pthread_mutex_unlock(mutexHandle);
		#endif
	}

	//

	inline SemaphoreHandle SemaphoreCreate(int initialCount, int maxCount)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			return ::CreateSemaphore(nullptr, initialCount, maxCount, nullptr);
		#else
			SemaphoreHandle semaphoreHandle = new sem_t();
			sem_init(semaphoreHandle, 0, initialCount);
			return semaphoreHandle;
		#endif
	}

	inline void SemaphoreDestroy(SemaphoreHandle semaphoreHandle)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			::CloseHandle(semaphoreHandle);
		#else
			sem_destroy(semaphoreHandle);
			delete semaphoreHandle;
		#endif
	}

	inline bool SemaphoreAcquire(SemaphoreHandle semaphoreHandle)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			int state = WaitForSingleObject(semaphoreHandle, INFINITE);
			ASSERT_RELEASE(state == WAIT_OBJECT_0 || state == WAIT_TIMEOUT);
			return state == WAIT_OBJECT_0;
		#else
			return sem_wait(semaphoreHandle) == 0;
		#endif
	}

	inline bool SemaphoreTryAcquire(SemaphoreHandle semaphoreHandle)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			int state = WaitForSingleObject(semaphoreHandle, 0);
			ASSERT_RELEASE(state == WAIT_OBJECT_0 || state == WAIT_TIMEOUT);
			return state == WAIT_OBJECT_0;
		#else
			return sem_trywait(semaphoreHandle) == 0;
		#endif
	}

	inline void SemaphoreRelease(SemaphoreHandle semaphoreHandle, int count)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			::ReleaseSemaphore(semaphoreHandle, count, nullptr);
		#else
			for (int i = 0; i < count; i++)
				sem_post(semaphoreHandle);
		#endif
	}

	//

	inline int32 AtomicIncrement32(int32* data)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			int32 newValue = ::InterlockedIncrement((long*)data);
		#else
			int32 newValue = __sync_add_and_fetch(data, 1);
		#endif

		return newValue - 1;
	}

	inline int32 AtomicDecrement32(int32* data)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			int32 newValue = ::InterlockedDecrement((long*)data);
		#else
			int32 newValue = __sync_sub_and_fetch(data, 1);
		#endif

		return newValue + 1;
	}

	inline int32 AtomicAdd32(int32* data, int x)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			return ::InterlockedExchangeAdd((long*)data, x);
		#else
			return __sync_fetch_and_add(data, x);
		#endif
	}

	inline int32 AtomicCompareExchange32(int32* data, int32 comparand, int32 exchange)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			return ::InterlockedCompareExchange((long*)data, exchange, comparand);
		#else
			return __sync_val_compare_and_swap(data, comparand, exchange);
		#endif
	}
}
