/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_MESH_CACHE
#define BE_SCENE_MESH_CACHE

#include "beScene.h"
#include <beCore/beShared.h>
#include <beCore/beResourceManagerImpl.h>
#include <lean/tags/noncopyable.h>
#include "beAssembledMesh.h"
#include <beGraphics/beDevice.h>
#include <beCore/bePathResolver.h>
#include <beCore/beContentProvider.h>
#include <beCore/beComponentMonitor.h>
#include <lean/pimpl/pimpl_ptr.h>

namespace beScene
{

/// Mesh cache base.
class MeshCache : public lean::noncopyable, public beCore::FiledResourceManagerImpl<AssembledMesh, MeshCache>
{
	friend ResourceManagerImpl;
	friend FiledResourceManagerImpl;

public:
	struct M;

private:
	lean::pimpl_ptr<M> m;

public:
		/// Constructor.
	BE_SCENE_API MeshCache(beGraphics::Device *device, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider);
	/// Destructor.
	BE_SCENE_API ~MeshCache();

	using FiledResourceManagerImpl::SetName;
	/// Adds a named mesh.
	BE_SCENE_API AssembledMesh* SetName(beScene::Mesh *mesh, const lean::utf8_ntri &name);

	/// Gets a mesh from the given file.
	BE_SCENE_API AssembledMesh* GetByFile(const lean::utf8_ntri &file);

	/// Commits / reacts to changes.
	BE_SCENE_API void Commit();

	/// Sets the component monitor.
	BE_SCENE_API void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor);
	/// Gets the component monitor.
	BE_SCENE_API beCore::ComponentMonitor* GetComponentMonitor() const;

	/// Gets the path resolver.
	BE_SCENE_API const beCore::PathResolver& GetPathResolver() const;
};

/// Creates a new mesh cache.
BE_SCENE_API lean::resource_ptr<MeshCache, true> CreateMeshCache(beGraphics::Device *device,
	const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider);

} // namespace

#endif