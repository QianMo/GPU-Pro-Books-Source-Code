/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_TEXTURE_CACHE_DX11
#define BE_GRAPHICS_TEXTURE_CACHE_DX11

#include "beGraphics.h"
#include "../beTextureCache.h"
#include <beCore/beResourceManagerImpl.h>
#include "beD3D11.h"
#include <lean/pimpl/pimpl_ptr.h>
#include <lean/smart/resource_ptr.h>

namespace beGraphics
{

namespace DX11
{

/// Texture cache implementation.
class TextureCache : public beCore::FiledResourceManagerImpl<beGraphics::Texture, TextureCache, beGraphics::TextureCache>
{
	friend ResourceManagerImpl;
	friend FiledResourceManagerImpl;

public:
	struct M;

private:
	lean::pimpl_ptr<M> m;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API TextureCache(api::Device *pDevice, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~TextureCache();

	/// Gets a texture from the given file.
	BE_GRAPHICS_DX11_API beGraphics::Texture* GetByFile(const lean::utf8_ntri &file, bool bSRGB = false) LEAN_OVERRIDE;
	
	/// Gets a texture for the given texture view.
	BE_GRAPHICS_DX11_API beGraphics::Texture* GetTexture(const beGraphics::TextureView *pTexture) const LEAN_OVERRIDE;
	/// Gets a texture view for the given texture.
	BE_GRAPHICS_DX11_API beGraphics::TextureView* GetView(const beGraphics::Texture *pTexture) LEAN_OVERRIDE;

	/// Gets whether the given texture is an srgb texture.
	BE_GRAPHICS_DX11_API bool IsSRGB(const beGraphics::Texture *pTexture) const;

	/// Commits changes / reacts to changes.
	BE_GRAPHICS_DX11_API void Commit();

	/// Sets the component monitor.
	BE_GRAPHICS_DX11_API void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor) LEAN_OVERRIDE;
	/// Gets the component monitor.
	BE_GRAPHICS_DX11_API beCore::ComponentMonitor* GetComponentMonitor() const LEAN_OVERRIDE;

	/// Gets the path resolver.
	BE_GRAPHICS_DX11_API const beCore::PathResolver& GetPathResolver() const LEAN_OVERRIDE;

	/// Gets the implementation identifier.
	ImplementationID GetImplementationID() const LEAN_OVERRIDE { return DX11Implementation; }
};

template <> struct ToImplementationDX11<beGraphics::TextureCache> { typedef TextureCache Type; };

} // namespace

} // namespace

#endif