/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_MATERIAL_CACHE
#define BE_GRAPHICS_MATERIAL_CACHE

#include "beGraphics.h"
#include <beCore/beShared.h>
#include <beCore/beResourceManager.h>
#include <lean/tags/noncopyable.h>
#include "beMaterial.h"
#include <beCore/bePathResolver.h>
#include <beCore/beContentProvider.h>
#include <beCore/beComponentMonitor.h>
#include <lean/smart/resource_ptr.h>

namespace beGraphics
{

class EffectCache;
class MaterialConfigCache;

/// Material configuration cache.
class LEAN_INTERFACE MaterialCache : public lean::noncopyable, public beCore::ResourceManager<Material>, public Implementation
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(MaterialCache)

public:
	/// Gets a material from the given file.
	virtual Material* NewByFile(const lean::utf8_ntri &file, const lean::utf8_ntri &name = "") = 0;

	/// Sets the component monitor.
	virtual void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor) = 0;
	/// Gets the component monitor.
	virtual beCore::ComponentMonitor* GetComponentMonitor() const = 0;

	/// Gets the path resolver.
	virtual const beCore::PathResolver& GetPathResolver() const = 0;
};

/// Creates a new texture cache.
BE_GRAPHICS_API lean::resource_ptr<MaterialCache, lean::critical_ref>
	CreateMaterialCache(EffectCache *effectCache, MaterialConfigCache *configCache,
	const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider);

} // namespace

#endif