/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beResourceManager.h"
#include <beCore/beFileSystem.h>

#include <beCore/beFileSystemPathResolver.h>
#include <beCore/beFileContentProvider.h>

namespace beScene
{

namespace
{

/// Creates an effect cache.
lean::resource_ptr<beGraphics::EffectCache, true> CreateEffectCache(beGraphics::Device *pDevice,
	const utf8_ntri &effectCacheLocation, const utf8_ntri &effectLocation, beGraphics::TextureCache *pTextureCache)
{
	return beGraphics::CreateEffectCache(*pDevice, pTextureCache,
		beCore::FileSystem::Get().GetPrimaryPath(effectCacheLocation, true),
		beCore::FileSystemPathResolver(effectLocation), beCore::FileContentProvider() );
}

/// Creates a texture cache.
lean::resource_ptr<beGraphics::TextureCache, true> CreateTextureCache(beGraphics::Device *pDevice,
	const utf8_ntri &textureLocation)
{
	return beGraphics::CreateTextureCache(*pDevice,
		beCore::FileSystemPathResolver(textureLocation), beCore::FileContentProvider() );
}

/// Creates a material cache.
lean::resource_ptr<beGraphics::MaterialConfigCache, true> CreateMaterialConfigCache(beGraphics::TextureCache *pTextureCache, const utf8_ntri &materialLocation)
{
	return beg::CreateMaterialConfigCache(pTextureCache,
		beCore::FileSystemPathResolver(materialLocation), beCore::FileContentProvider() );
}

/// Creates a material cache.
lean::resource_ptr<beGraphics::MaterialCache, true> CreateMaterialCache(beGraphics::EffectCache *pEffectCache, beGraphics::MaterialConfigCache *pConfigCache,
	const utf8_ntri &materialLocation)
{
	return beg::CreateMaterialCache(pEffectCache, pConfigCache,
		beCore::FileSystemPathResolver(materialLocation), beCore::FileContentProvider() );
}

/// Creates a mesh cache.
lean::resource_ptr<MeshCache, true> CreateMeshCache(beGraphics::Device *device,
	const utf8_ntri &meshLocation)
{
	return beScene::CreateMeshCache(device,
		beCore::FileSystemPathResolver(meshLocation), beCore::FileContentProvider() );
}

} // namespace

// Constructor.
ResourceManager::ResourceManager(beCore::ComponentMonitor *monitor, beGraphics::EffectCache *effectCache, beGraphics::TextureCache *textureCache,
	beGraphics::MaterialConfigCache *materialConfigCache, beGraphics::MaterialCache *materialCache, class MeshCache *meshCache)
	: Monitor( LEAN_ASSERT_NOT_NULL(monitor) ),
	EffectCache( LEAN_ASSERT_NOT_NULL(effectCache) ),
	TextureCache( LEAN_ASSERT_NOT_NULL(textureCache) ),
	MaterialConfigCache( LEAN_ASSERT_NOT_NULL(materialConfigCache) ),
	MaterialCache( LEAN_ASSERT_NOT_NULL(materialCache) ),
	MeshCache( LEAN_ASSERT_NOT_NULL(meshCache) )
{
}

// Copy constructor.
ResourceManager::ResourceManager(const ResourceManager &right)
	// MONITOR: REDUNDANT
	: Monitor( right.Monitor ),
	EffectCache( right.EffectCache ),
	TextureCache( right.TextureCache ),
	MaterialCache( right.MaterialCache ),
	MeshCache( right.MeshCache )
{
}

// Destructor.
ResourceManager::~ResourceManager()
{
}

// Notifies dependent listeners about dependency changes.
void ResourceManager::Commit()
{
	// TODO: Process call here?
	Monitor->Process();

	EffectCache->Commit();
	TextureCache->Commit();
	MaterialConfigCache->Commit();
	MaterialCache->Commit();
	MeshCache->Commit();
}

// Creates a resource manager from the given device.
lean::resource_ptr<ResourceManager, true> CreateResourceManager(beGraphics::Device *device,
	const utf8_ntri &effectCacheDir, const utf8_ntri &effectDir, const utf8_ntri &textureDir, const utf8_ntri &materialDir, const utf8_ntri &meshDir,
	beCore::ComponentMonitor *pMonitor)
{
	LEAN_ASSERT(device != nullptr);

	lean::resource_ptr<beCore::ComponentMonitor> monitor = pMonitor;
	if (!pMonitor)
		monitor = new_resource bec::ComponentMonitor();

	lean::resource_ptr<beGraphics::TextureCache> textureCache = CreateTextureCache(device, textureDir);
	lean::resource_ptr<beGraphics::EffectCache> effectCache = CreateEffectCache(device, effectCacheDir, effectDir, textureCache);
	lean::resource_ptr<beGraphics::MaterialConfigCache> materialConfigCache = CreateMaterialConfigCache(textureCache, materialDir);
	lean::resource_ptr<beGraphics::MaterialCache> materialCache = CreateMaterialCache(effectCache, materialConfigCache, materialDir);
	lean::resource_ptr<MeshCache> meshCache = CreateMeshCache(device, meshDir);
	
	effectCache->SetComponentMonitor(monitor);
	textureCache->SetComponentMonitor(monitor);
	materialConfigCache->SetComponentMonitor(monitor);
	materialCache->SetComponentMonitor(monitor);
	meshCache->SetComponentMonitor(monitor);

	return CreateResourceManager(effectCache, textureCache, materialConfigCache, materialCache, meshCache, monitor);
}

// Creates a resource manager from the given effect cache.
lean::resource_ptr<ResourceManager, true> CreateResourceManager(beGraphics::EffectCache *effectCache, beGraphics::TextureCache *textureCache,
	beGraphics::MaterialConfigCache *materialConfigCache, beGraphics::MaterialCache *materialCache, MeshCache *meshCache,
	beCore::ComponentMonitor *pMonitor)
{
	return new_resource ResourceManager(
			(pMonitor) ? pMonitor : (new_resource bec::ComponentMonitor()).get(),
			effectCache, textureCache, materialConfigCache, materialCache, meshCache
		);
}

} // namespace
