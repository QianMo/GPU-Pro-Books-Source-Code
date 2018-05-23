/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_SERIALIZATION_TASKS
#define BE_ENTITYSYSTEM_SERIALIZATION_TASKS

#include "beEntitySystem.h"
#include <beCore/beSerializationJobs.h>
#include <beCore/beParameterSet.h>

namespace beEntitySystem
{

/// Gets generic tasks to perform when saving.
BE_ENTITYSYSTEM_API beCore::SaveJobs& GetWorldSaveTasks();
/// Gets generic tasks to perform when loading.
BE_ENTITYSYSTEM_API beCore::LoadJobs& GetWorldLoadTasks();

/// Gets generic tasks to perform when saving.
BE_ENTITYSYSTEM_API beCore::SaveJobs& GetResourceSaveTasks();
/// Gets generic tasks to perform when loading.
BE_ENTITYSYSTEM_API beCore::LoadJobs& GetResourceLoadTasks();

/// Gets the serialization parameter layout.
BE_ENTITYSYSTEM_API beCore::ParameterLayout& GetSerializationParameters();

/// Gets the load job stored under the given name, creates and stores it if missing.
template <class Job, class Token, class Parameters>
Job& GetOrMakeLoadJob(Parameters &parameters, const utf8_ntri &name, beCore::SerializationQueue<beCore::LoadJob> &queue)
{
	beCore::ParameterLayout &layout = GetSerializationParameters();
	static const uint4 parameterID = layout.Add(name);
	return *beCore::GetOrMake<Job*>(parameters, layout, parameterID, beCore::LoadJobFactory<Job>(queue));
}

/// Gets the save job stored under the given name, creates and stores it if missing.
template <class Job, class Token, class Parameters>
Job& GetOrMakeSaveJob(Parameters &parameters, const utf8_ntri &name, beCore::SerializationQueue<beCore::SaveJob> &queue)
{
	beCore::ParameterLayout &layout = GetSerializationParameters();
	static const uint4 parameterID = layout.Add(name);
	return *beCore::GetOrMake<Job*>(parameters, layout, parameterID, beCore::SaveJobFactory<Job>(queue));
}

} // namespace

#endif