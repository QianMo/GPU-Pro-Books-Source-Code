/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERABLE_MESH_CACHE
#define BE_SCENE_RENDERABLE_MESH_CACHE

#include "beScene.h"
#include <beCore/beShared.h>
#include <beCore/beResourceManagerImpl.h>
#include <lean/tags/noncopyable.h>
#include "beRenderableMesh.h"
#include <beGraphics/beDevice.h>
#include <beCore/beComponentMonitor.h>
#include <lean/pimpl/pimpl_ptr.h>

namespace beScene
{

/// Mesh cache base.
class RenderableMeshCache : public lean::noncopyable, public beCore::ResourceManagerImpl<RenderableMesh, RenderableMeshCache>
{
	friend ResourceManagerImpl;

public:
	struct M;

private:
	lean::pimpl_ptr<M> m;

public:
	/// Constructor.
	BE_SCENE_API RenderableMeshCache(beGraphics::Device *device);
	/// Destructor.
	BE_SCENE_API ~RenderableMeshCache();

	/// Commits / reacts to changes.
	BE_SCENE_API void Commit() LEAN_OVERRIDE;

	/// Sets the component monitor.
	BE_SCENE_API void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor);
	/// Gets the component monitor.
	BE_SCENE_API beCore::ComponentMonitor* GetComponentMonitor() const;
};

/// Creates a new mesh cache.
BE_SCENE_API lean::resource_ptr<RenderableMeshCache, true> CreateRenderableMeshCache(beGraphics::Device *device);

} // namespace

#endif