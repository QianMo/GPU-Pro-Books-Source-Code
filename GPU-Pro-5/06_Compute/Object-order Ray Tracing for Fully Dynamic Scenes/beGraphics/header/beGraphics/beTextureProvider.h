/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_TEXTURE_PROVIDER
#define BE_GRAPHICS_TEXTURE_PROVIDER

#include "beGraphics.h"
#include "beTexture.h"

namespace beGraphics
{

/// Generic texture provider base class.
class TextureProvider
{
protected:
	TextureProvider& operator =(const TextureProvider&) { return *this; }
	~TextureProvider() throw() { }

public:
	/// Gets the number of textures.
	virtual uint4 GetTextureCount() const = 0;
	/// Gets the ID of the given texture.
	virtual uint4 GetTextureID(const utf8_ntri &name) const = 0;
	/// Gets the name of the given texture.
	virtual utf8_ntr GetTextureName(uint4 id) const = 0;
	/// Gets whether the texture is a color texture.
	virtual bool IsColorTexture(uint4 id) const = 0;

	/// Sets the given texture.
	virtual void SetTexture(uint4 id, const TextureView *pView) = 0;
	/// Gets the given texture.
	virtual const TextureView* GetTexture(uint4 id) const = 0;
};

/// Enhanced generic texture provider base class.
class EnhancedTextureProvider : public TextureProvider
{
protected:
	EnhancedTextureProvider& operator =(const EnhancedTextureProvider&) { return *this; }
	~EnhancedTextureProvider() throw() { }

public:
	/// Resets the given texture to its default value.
	virtual bool ResetTexture(uint4 id) = 0;

	/// Gets a default value provider.
	virtual const TextureProvider* GetTextureDefaults() const = 0;
};

/// Transfers all textures from the given source texture provider to the given destination texture provider.
BE_GRAPHICS_API void TransferTextures(TextureProvider &dest, const TextureProvider &source);

} // namespace

#endif