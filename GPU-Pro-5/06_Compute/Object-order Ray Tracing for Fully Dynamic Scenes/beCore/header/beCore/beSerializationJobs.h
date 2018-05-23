/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_SERIALIZATION_JOB
#define BE_CORE_SERIALIZATION_JOB

#include "beCore.h"
#include "beShared.h"
#include <lean/tags/noncopyable.h>
#include <lean/rapidxml/rapidxml.hpp>
#include <lean/pimpl/pimpl_ptr.h>

namespace beCore
{
	
class ParameterSet;

/// Serialization queue.
template <class Job>
class LEAN_INTERFACE SerializationQueue
{
	LEAN_INTERFACE_BEHAVIOR(SerializationQueue)

public:
	/// Takes ownership of the given serialization job.
	virtual void AddSerializationJob(const Job *pJob) = 0;
};

/// Serialization job interface.
class LEAN_INTERFACE SaveJob : public lean::noncopyable_chain<Shared>
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(SaveJob)

public:
	/// Saves anything, e.g. to the given XML root node.
	virtual void Save(rapidxml::xml_node<lean::utf8_t> &root, ParameterSet &parameters, SerializationQueue<SaveJob> &queue) const = 0;
};

/// Serialization job interface.
class LEAN_INTERFACE LoadJob : public lean::noncopyable_chain<Shared>
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(LoadJob)

public:
	/// Loads anything, e.g from the given XML root node.
	virtual void Load(const rapidxml::xml_node<lean::utf8_t> &root, ParameterSet &parameters, SerializationQueue<LoadJob> &queue) const = 0;
};

/// Serialization queue.
class SaveJobs : public lean::noncopyable, public SerializationQueue<SaveJob>
{
public:
	struct M;

private:
	lean::pimpl_ptr<M> m;

public:
	/// Constructor.
	BE_CORE_API SaveJobs();
	/// Destructor.
	BE_CORE_API ~SaveJobs();

	/// Takes ownership of the given serialization job.
	BE_CORE_API void AddSerializationJob(const SaveJob *pJob);

	/// Saves anything, e.g. to the given XML root node.
	BE_CORE_API void Save(rapidxml::xml_node<lean::utf8_t> &root, ParameterSet &parameters);
};

/// Serialization queue.
class LoadJobs : public lean::noncopyable, public SerializationQueue<LoadJob>
{
public:
	struct M;

private:
	lean::pimpl_ptr<M> m;

public:
	/// Constructor.
	BE_CORE_API LoadJobs();
	/// Destructor.
	BE_CORE_API ~LoadJobs();

	/// Takes ownership of the given serialization job.
	BE_CORE_API void AddSerializationJob(const LoadJob *pJob);

	/// Loads anything, e.g from the given XML root node.
	BE_CORE_API void Load(const rapidxml::xml_node<lean::utf8_t> &root, ParameterSet &parameters);
};

/// Instantiate this to add a serializer of the given type to the global queue of save tasks.
template <class SaveTask, SaveJobs& (*GetSaveTasks)()>
struct SaveTaskPlugin
{
	/// Adds a global save task of the given type.
	SaveTaskPlugin()
	{
		GetSaveTasks().AddSerializationJob( new SaveTask() );
	}
};

/// Instantiate this to add a loader of the given type to the global queue of load tasks.
template <class LoadTask, LoadJobs& (*GetLoadTasks)()>
struct LoadTaskPlugin
{
	/// Adds a global load task of the given type.
	LoadTaskPlugin()
	{
		GetLoadTasks().AddSerializationJob( new LoadTask() );
	}
};

/// Save job factory.
template <class Job, class JobBase = beCore::SaveJob>
struct SaveJobFactory
{
	beCore::SerializationQueue<JobBase> &queue;
	
	/// Constructor.
	SaveJobFactory(beCore::SerializationQueue<JobBase> &queue)
		: queue(queue) { }

	/// Creates and queues a new job.
	Job* operator ()(...) const
	{
		// NOTE: Serialization queue takes ownership
		Job *job = new Job();
		queue.AddSerializationJob(job);
		return job;
	}
};

/// Load job factory.
template <class Job, class JobBase = beCore::LoadJob>
struct LoadJobFactory : SaveJobFactory<Job, JobBase>
{
	/// Constructor.
	LoadJobFactory(beCore::SerializationQueue<JobBase> &queue)
		: typename LoadJobFactory::SaveJobFactory(queue) { }
};

} // namespace

#endif