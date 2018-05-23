/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RESOURCE_MANAGER
#define BE_SCENE_RESOURCE_MANAGER

#include "beScene.h"
#include <beCore/beShared.h>
#include <beCore/beComponentMonitor.h>
#include <beGraphics/beEffectCache.h>
#include <beGraphics/beTextureCache.h>
#include <beGraphics/beMaterialCache.h>
#include <beGraphics/beMaterialConfigCache.h>
#include "beMeshCache.h"
#include <lean/smart/resource_ptr.h>

namespace beScene
{

/// Resource Manager class.
class ResourceManager : public beCore::Resource
{
public:
	/// Constructor.
	BE_SCENE_API ResourceManager(beCore::ComponentMonitor *monitor, beGraphics::EffectCache *effectCache, beGraphics::TextureCache *textureCache,
		beGraphics::MaterialConfigCache *materialConfigCache, beGraphics::MaterialCache *materialCache, MeshCache *meshCache);
	/// Copy constructor.
	BE_SCENE_API ResourceManager(const ResourceManager &right);
	/// Destructor.
	BE_SCENE_API ~ResourceManager();

	/// Monitor.
	lean::resource_ptr<beCore::ComponentMonitor> Monitor;

	/// Effect cache.
	lean::resource_ptr<beGraphics::EffectCache> EffectCache;

	/// Texture cache.
	lean::resource_ptr<beGraphics::TextureCache> TextureCache;

	/// Material configuration cache.
	lean::resource_ptr<beGraphics::MaterialConfigCache> MaterialConfigCache;
	/// Material cache.
	lean::resource_ptr<beGraphics::MaterialCache> MaterialCache;

	/// Mesh cache.
	lean::resource_ptr<MeshCache> MeshCache;

	/// Commits / reacts to changes.
	BE_SCENE_API void Commit();
};

/// Creates a resource manager from the given device.
BE_SCENE_API lean::resource_ptr<ResourceManager, true> CreateResourceManager(beGraphics::Device *device,
	const utf8_ntri &effectCacheDir, const utf8_ntri &effectDir, const utf8_ntri &textureDir, const utf8_ntri &materialDir, const utf8_ntri &meshDir,
	beCore::ComponentMonitor *pMonitor = nullptr);
/// Creates a resource manager from the given caches.
BE_SCENE_API lean::resource_ptr<ResourceManager, true> CreateResourceManager(
	beGraphics::EffectCache *effectCache, beGraphics::TextureCache *textureCache,
	beGraphics::MaterialConfigCache *materialConfigCache, beGraphics::MaterialCache *materialCache, MeshCache *meshCache,
	beCore::ComponentMonitor *pMonitor = nullptr);

} // namespace

#endif