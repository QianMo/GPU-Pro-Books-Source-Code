/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_TEXTURE_CACHE
#define BE_GRAPHICS_TEXTURE_CACHE

#include "beGraphics.h"
#include <beCore/beShared.h>
#include <beCore/beResourceManager.h>
#include <lean/tags/noncopyable.h>
#include "beTexture.h"
#include <beCore/bePathResolver.h>
#include <beCore/beContentProvider.h>
#include <beCore/beComponentMonitor.h>
#include <lean/smart/resource_ptr.h>

namespace beGraphics
{

/// Texture cache.
class LEAN_INTERFACE TextureCache : public lean::noncopyable, public beCore::FiledResourceManager<Texture>, public Implementation
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(TextureCache)

public:
	/// Gets a texture from the given file.
	virtual Texture* GetByFile(const lean::utf8_ntri &file, bool bSRGB = false) = 0;
	/// Gets a texture view from the given file.
	LEAN_INLINE TextureView* GetViewByFile(const lean::utf8_ntri &file, bool bSRGB = false)
	{
		return GetView( GetByFile(file, bSRGB) ); 
	}

	/// Gets a texture for the given texture view.
	virtual Texture* GetTexture(const TextureView *pTexture) const = 0;
	/// Gets a texture view for the given texture.
	virtual TextureView* GetView(const Texture *pTexture) = 0;

	/// Gets whether the given texture is an srgb texture.
	virtual bool IsSRGB(const Texture *pTexture) const = 0;

	/// Sets the component monitor.
	virtual void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor) = 0;
	/// Gets the component monitor.
	virtual beCore::ComponentMonitor* GetComponentMonitor() const = 0;

	/// Gets the path resolver.
	virtual const beCore::PathResolver& GetPathResolver() const = 0;
};

// Prototypes
class Device;

/// Creates a new texture cache.
BE_GRAPHICS_API lean::resource_ptr<TextureCache, true> CreateTextureCache(const Device &device, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider);

} // namespace

#endif