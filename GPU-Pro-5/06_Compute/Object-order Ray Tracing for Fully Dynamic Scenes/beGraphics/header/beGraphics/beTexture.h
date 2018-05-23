/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_TEXTURE
#define BE_GRAPHICS_TEXTURE

#include "beGraphics.h"
#include "beFormat.h"
#include <beCore/beShared.h>
#include <beCore/beManagedResource.h>
#include <beCore/beComponent.h>
#include <beGraphics/beDevice.h>
#include <beCore/beOpaqueHandle.h>
#include <lean/smart/resource_ptr.h>
#include <lean/tags/noncopyable.h>

namespace beGraphics
{

/// Texture description.
struct TextureDesc
{
	uint4 Width;			///< Width in pixels.
	uint4 Height;			///< Height in pixels.
	uint4 Depth;			///< Depth in pixels.
	Format::T Format;		///< Pixel format.
	uint4 MipLevels;		///< Number of mip levels.

	/// Constructor.
	explicit TextureDesc(uint4 width = 0,
		uint4 height = 0,
		uint4 depth = 0,
		Format::T format = Format::Unknown,
		uint4 mipLevels = 0)
			: Width(width),
			Height(height),
			Depth(depth),
			Format(format),
			MipLevels(mipLevels) { }
};

/// Texture type enumeration.
struct TextureType
{
	/// Enumeration.
	enum T
	{
		Texture1D,		///< 1D texture.
		Texture2D,		///< 2D texture.
		Texture3D,		///< 3D texture.

		NotATexture		///< Not a texture.
	};
	LEAN_MAKE_ENUM_STRUCT(TextureType)
};

class TextureCache;

/// Texture resource interface.
class LEAN_INTERFACE Texture : public lean::nonassignable, public beCore::OptionalResource,
	public beCore::ManagedResource<TextureCache>, public beCore::HotResource<Texture>, public Implementation
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(Texture)

public:
	/// Gets the texture description.
	virtual TextureDesc GetDesc() const = 0;
	/// Gets the texture type.
	virtual TextureType::T GetType() const = 0;
};

/// Texture view interface.
class LEAN_INTERFACE TextureView : public beCore::OptionalResource,
	public beCore::ManagedResource<TextureCache>, public beCore::HotResource<TextureView>, public Implementation
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(TextureView)

public:
	/// Gets the texture description.
	virtual TextureDesc GetDesc() const = 0;
	/// Gets the texture type.
	virtual TextureType::T GetType() const = 0;

	/// Gets the component type.
	BE_GRAPHICS_API static const beCore::ComponentType* GetComponentType();
};

/// Texture view handle.
typedef beCore::OpaqueHandle<TextureView> TextureViewHandle;
using beCore::ToImpl;


// Prototypes
class SwapChain;

/// Loads a texture from the given file.
BE_GRAPHICS_API lean::resource_ptr<Texture, true> LoadTexture(const Device &device, const lean::utf8_ntri &fileName, const TextureDesc *pDesc = nullptr, bool bSRGB = false);
/// Loads a texture from the given memory.
BE_GRAPHICS_API lean::resource_ptr<Texture, true> LoadTexture(const Device &device, const char *data, uint4 dataLength, const TextureDesc *pDesc = nullptr, bool bSRGB = false);
/// Creates a texture view from the given texture.
BE_GRAPHICS_API lean::resource_ptr<TextureView, true> ViewTexture(const Texture &texture, const Device &device);

/// Gets the back buffer.
BE_GRAPHICS_API lean::resource_ptr<Texture, true> GetBackBuffer(const SwapChain &swapChain, uint4 index = 0);

} // namespace

#endif