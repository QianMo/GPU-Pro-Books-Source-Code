#pragma once


#include <essentials/stl.h>
#include <system/threads.h>


using namespace NEssentials;
using namespace NSystem;


namespace NCommon
{
	class Job;
	class JobGroup;
	class JobSystem;


	class Job
	{
	public:
		Job()
		{
			done = false;
			owner = nullptr;
		}
		virtual ~Job() {}

		virtual bool Do() = 0;

	public:
		bool done;
		JobGroup* owner;
	};


	class JobGroup
	{
	public:
		JobGroup()
		{
			mutex = MutexCreate();
			jobsCount = 0;
			jobsDoneCount = 0;
		}
		~JobGroup()
		{
			MutexDestroy(mutex);
		}

		void AddJob(Job* job);
		bool JobsDone();
		void Wait();

	public:
		MutexHandle mutex;
		int32 jobsCount;
		int32 jobsDoneCount;
		vector<Job*> jobs;
	};


	class JobSystem
	{
	public:
		void Create(int threadsCount);
		void Destroy();

		void AddJob(Job* job);
		void AddJobGroup(const JobGroup& jobGroup);

	public:
		SemaphoreHandle activeJobsSemaphore;
		MutexHandle jobsQueueMutex;
		vector<ThreadHandle> threads;
		queue<Job*> jobs;
	};
}
