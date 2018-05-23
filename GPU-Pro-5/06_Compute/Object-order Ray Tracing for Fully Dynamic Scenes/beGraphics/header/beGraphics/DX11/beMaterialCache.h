/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_MATERIAL_CACHE_DX11
#define BE_GRAPHICS_MATERIAL_CACHE_DX11

#include "beGraphics.h"
#include "../beMaterialCache.h"
#include <beCore/beResourceManagerImpl.h>
#include <D3D11.h>
#include <lean/pimpl/pimpl_ptr.h>
#include <lean/smart/resource_ptr.h>

namespace beGraphics
{

namespace DX11
{

class EffectCache;
class MaterialConfigCache;

/// Material configuration cache implementation.
class MaterialCache : public beCore::ResourceManagerImpl<beGraphics::Material, MaterialCache, beGraphics::MaterialCache>
{
	friend ResourceManagerImpl;
//	friend FiledResourceManagerImpl;

public:
	struct M;

private:
	lean::pimpl_ptr<M> m;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API MaterialCache(EffectCache *effectCache, MaterialConfigCache *configCache, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~MaterialCache();
	
	/// Gets a material from the given file.
	BE_GRAPHICS_DX11_API beGraphics::Material* NewByFile(const lean::utf8_ntri &file, const lean::utf8_ntri &name = "") LEAN_OVERRIDE;
	
	/// Commits changes / reacts to changes.
	BE_GRAPHICS_DX11_API void Commit() LEAN_OVERRIDE;

	/// Sets the component monitor.
	BE_GRAPHICS_DX11_API void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor) LEAN_OVERRIDE;
	/// Gets the component monitor.
	BE_GRAPHICS_DX11_API beCore::ComponentMonitor* GetComponentMonitor() const LEAN_OVERRIDE;

	/// Gets the path resolver.
	BE_GRAPHICS_DX11_API const beCore::PathResolver& GetPathResolver() const LEAN_OVERRIDE;

	/// Gets the implementation identifier.
	ImplementationID GetImplementationID() const LEAN_OVERRIDE { return DX11Implementation; }
};

template <> struct ToImplementationDX11<beGraphics::MaterialCache> { typedef MaterialCache Type; };

} // namespace

} // namespace

#endif