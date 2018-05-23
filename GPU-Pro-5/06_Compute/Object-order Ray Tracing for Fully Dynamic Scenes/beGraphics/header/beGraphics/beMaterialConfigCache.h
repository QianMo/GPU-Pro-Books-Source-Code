/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_MATERIAL_CONFIG_CACHE
#define BE_GRAPHICS_MATERIAL_CONFIG_CACHE

#include "beGraphics.h"
#include <beCore/beShared.h>
#include <beCore/beResourceManager.h>
#include <lean/tags/noncopyable.h>
#include "beMaterialConfig.h"
#include <beCore/bePathResolver.h>
#include <beCore/beContentProvider.h>
#include <beCore/beComponentMonitor.h>
#include <lean/smart/resource_ptr.h>

namespace beGraphics
{

class TextureCache;

/// Material configuration cache.
class LEAN_INTERFACE MaterialConfigCache : public lean::noncopyable, public beCore::ResourceManager<MaterialConfig>, public Implementation
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(MaterialConfigCache)

public:
/*	/// Gets a material configuration from the given file.
	virtual MaterialConfig* GetByFile(const lean::utf8_ntri &file) = 0;
*/
	/// Sets the component monitor.
	virtual void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor) = 0;
	/// Gets the component monitor.
	virtual beCore::ComponentMonitor* GetComponentMonitor() const = 0;

	/// Gets the texture cache.
	virtual TextureCache* GetTextureCache() const = 0;
	/// Gets the path resolver.
	virtual const beCore::PathResolver& GetPathResolver() const = 0;
};

/// Creates a new texture cache.
BE_GRAPHICS_API lean::resource_ptr<MaterialConfigCache, lean::critical_ref>
	CreateMaterialConfigCache(TextureCache *textureCache, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider);

} // namespace

#endif