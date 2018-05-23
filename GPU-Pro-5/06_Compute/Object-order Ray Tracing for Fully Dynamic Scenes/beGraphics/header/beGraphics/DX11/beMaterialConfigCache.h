/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_MATERIAL_CONFIG_CACHE_DX11
#define BE_GRAPHICS_MATERIAL_CONFIG_CACHE_DX11

#include "beGraphics.h"
#include "../beMaterialConfigCache.h"
#include <beCore/beResourceManagerImpl.h>
#include <D3D11.h>
#include <lean/pimpl/pimpl_ptr.h>
#include <lean/smart/resource_ptr.h>

namespace beGraphics
{

namespace DX11
{

class TextureCache;

/// Material configuration cache implementation.
class MaterialConfigCache : public beCore::ResourceManagerImpl<beGraphics::MaterialConfig, MaterialConfigCache, beGraphics::MaterialConfigCache>
{
	friend ResourceManagerImpl;
//	friend FiledResourceManagerImpl;

public:
	struct M;

private:
	lean::pimpl_ptr<M> m;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API MaterialConfigCache(TextureCache *textureCache, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~MaterialConfigCache();
	
	/// Commits changes / reacts to changes.
	BE_GRAPHICS_DX11_API void Commit() LEAN_OVERRIDE;

/*	/// Gets a texture from the given file.
	BE_GRAPHICS_DX11_API beGraphics::MaterialConfig* GetByFile(const lean::utf8_ntri &file) LEAN_OVERRIDE;
*/
	/// Sets the component monitor.
	BE_GRAPHICS_DX11_API void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor) LEAN_OVERRIDE;
	/// Gets the component monitor.
	BE_GRAPHICS_DX11_API beCore::ComponentMonitor* GetComponentMonitor() const LEAN_OVERRIDE;

	/// Gets the texture cache.
	BE_GRAPHICS_DX11_API beGraphics::TextureCache* GetTextureCache() const LEAN_OVERRIDE;
	/// Gets the path resolver.
	BE_GRAPHICS_DX11_API const beCore::PathResolver& GetPathResolver() const LEAN_OVERRIDE;

	/// Gets the implementation identifier.
	ImplementationID GetImplementationID() const LEAN_OVERRIDE { return DX11Implementation; }
};

template <> struct ToImplementationDX11<beGraphics::MaterialConfigCache> { typedef MaterialConfigCache Type; };

} // namespace

} // namespace

#endif