/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"

#include <beEntitySystem/beSerializationTasks.h>

namespace beScene
{

const bec::LoadJob *CreateMaterialConfigLoader();
const bec::LoadJob *CreateMaterialLoader();
const bec::LoadJob *CreateMeshImportLoader();
const bec::LoadJob *CreateMeshLoader();

namespace
{

struct LoadTaskPlugin
{
	LoadTaskPlugin()
	{
		// ORDER: Loaders are interdependent
		bec::LoadJobs &jobs = beEntitySystem::GetResourceLoadTasks();
		jobs.AddSerializationJob( CreateMaterialConfigLoader() );
		jobs.AddSerializationJob( CreateMaterialLoader() );
		jobs.AddSerializationJob( CreateMeshImportLoader() );
		jobs.AddSerializationJob( CreateMeshLoader() );
	}

} LoadTaskPlugin;

} // namespace

} // namespace
