/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beSerializationTasks.h"

namespace beEntitySystem
{

// Gets generic tasks to perform when saving.
beCore::SaveJobs& GetWorldSaveTasks()
{
	static beCore::SaveJobs saveTasks;
	return saveTasks;
}

// Gets generic tasks to perform when loading.
beCore::LoadJobs& GetWorldLoadTasks()
{
	static beCore::LoadJobs loadTasks;
	return loadTasks;
}

// Gets generic tasks to perform when saving.
beCore::SaveJobs& GetResourceSaveTasks()
{
	static beCore::SaveJobs saveTasks;
	return saveTasks;
}

// Gets generic tasks to perform when loading.
beCore::LoadJobs& GetResourceLoadTasks()
{
	static beCore::LoadJobs loadTasks;
	return loadTasks;
}

} // namespace
