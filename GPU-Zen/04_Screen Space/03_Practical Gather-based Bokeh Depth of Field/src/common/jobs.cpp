#include <common/jobs.h>


using namespace NSystem;


//


void NCommon::JobGroup::AddJob(Job* job)
{
	job->owner = this;
	jobsCount++;
	jobs.push_back(job);
}


bool NCommon::JobGroup::JobsDone()
{
	MutexLock(mutex);

	if (jobsCount == jobsDoneCount)
	{
		MutexUnlock(mutex);
		return true;
	}
	else
	{
		MutexUnlock(mutex);
		return false;
	}
}


void NCommon::JobGroup::Wait()
{
	while (!JobsDone())
	{
	}
}


//


#ifdef MAXEST_FRAMEWORK_WINDOWS
	ThreadEntryReturnValue _stdcall JobThread(void* data)
#else
	ThreadEntryReturnValue JobThread(void* data)
#endif
{
	NCommon::JobSystem* jobSystem = (NCommon::JobSystem*)data;

	for (;;)
	{
		SemaphoreAcquire(jobSystem->activeJobsSemaphore);

		MutexLock(jobSystem->jobsQueueMutex);
		NCommon::Job* job = jobSystem->jobs.front();
		jobSystem->jobs.pop();
		MutexUnlock(jobSystem->jobsQueueMutex);

		if (job->Do())
		{
			job->done = true;
			if (job->owner)
				AtomicIncrement32(&job->owner->jobsDoneCount);
		}
	}

	return 0;
}


void NCommon::JobSystem::Create(int threadsCount)
{
	activeJobsSemaphore = SemaphoreCreate(0, threadsCount);
	jobsQueueMutex = MutexCreate();

	for (int i = 0; i < threadsCount; i++)
	{
		ThreadHandle thread = ThreadCreate(JobThread, this);
		threads.push_back(thread);
	}
}


void NCommon::JobSystem::Destroy()
{
	for (uint i = 0; i < threads.size(); i++)
		ThreadDestroy(threads[i]);
	threads.clear();

	MutexDestroy(jobsQueueMutex);
	SemaphoreDestroy(activeJobsSemaphore);
}


void NCommon::JobSystem::AddJob(Job* job)
{
	MutexLock(jobsQueueMutex);
	jobs.push(job);
	MutexUnlock(jobsQueueMutex);

	SemaphoreRelease(activeJobsSemaphore, 1);
}


void NCommon::JobSystem::AddJobGroup(const JobGroup& jobGroup)
{
	MutexLock(jobsQueueMutex);
	for (uint i = 0; i < jobGroup.jobs.size(); i++)
		jobs.push(jobGroup.jobs[i]);
	MutexUnlock(jobsQueueMutex);

	SemaphoreRelease(activeJobsSemaphore, jobGroup.jobs.size());
}
