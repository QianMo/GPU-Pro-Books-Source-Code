/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_TEXTURE_TARGET_POOL_DX11
#define BE_GRAPHICS_TEXTURE_TARGET_POOL_DX11

#include "beGraphics.h"
#include "../beTextureTargetPool.h"
#include "beFormat.h"
#include <D3D11.h>
#include <lean/smart/com_ptr.h>
#include <lean/smart/scoped_ptr.h>
#include <lean/pimpl/pimpl_ptr.h>

namespace beGraphics
{

namespace DX11
{

/// Texture target description.
struct TextureTargetDesc
{
	uint4 Width;				///< Target width.
	uint4 Height;				///< Target height.
	uint4 MipLevels;			///< Mip level count. Set most significant bit for auto-generation.
	DXGI_FORMAT Format;			///< Pixel format.
	DXGI_SAMPLE_DESC Samples;	///< Multi-sampling options.
	uint4 Count;				///< Number of texture array elements.

	/// NON-INITIALIZING constructor.
	TextureTargetDesc() { }
	/// Constructor.
	TextureTargetDesc(uint4 width, uint4 height,
		uint4 mipLevels,
		DXGI_FORMAT format,
		const DXGI_SAMPLE_DESC &samples,
		uint4 count = 1)
			: Width(width),
			Height(height),
			MipLevels(mipLevels),
			Format(format),
			Samples(samples),
			Count(count) { }
};

/// Converts the description into a DX11 description.
LEAN_INLINE TextureTargetDesc ToAPI(const beGraphics::TextureTargetDesc& desc)
{
	return TextureTargetDesc(
		desc.Width,
		desc.Height,
		desc.MipLevels,
		ToAPI(desc.Format),
		ToAPI(desc.Samples),
		desc.Count);
}

/// Converts the description into a breeze description.
LEAN_INLINE beGraphics::TextureTargetDesc FromAPI(const TextureTargetDesc& desc)
{
	return beGraphics::TextureTargetDesc(
		desc.Width,
		desc.Height,
		desc.MipLevels,
		FromAPI(desc.Format),
		FromAPI(desc.Samples),
		desc.Count);
}

} // namespace

using DX11::ToAPI;

/// Texture target.
class TextureTarget
{
public:
	typedef lean::com_ptr<ID3D11ShaderResourceView> texture_ptr;
	typedef lean::scoped_ptr<texture_ptr[]> scoped_texture_array_ptr;

protected:
	DX11::TextureTargetDesc m_desc;

	lean::com_ptr<ID3D11Resource> m_pResource;

	lean::com_ptr<ID3D11ShaderResourceView> m_pTexture;

	scoped_texture_array_ptr m_pTextures;

	mutable int m_references;
	mutable int m_uses;

	/// Constructor. Texture is OPTIONAL.
	BE_GRAPHICS_API TextureTarget(const DX11::TextureTargetDesc &desc, ID3D11Resource *pResource, ID3D11ShaderResourceView *pTexture, texture_ptr *pTextures = nullptr);
	/// Destructor.
	BE_GRAPHICS_API ~TextureTarget();

public:
	/// Gets the resource.
	LEAN_INLINE ID3D11Resource* GetResource() const { return m_pResource; }
	/// Gets the texture. OPTIONAL, may return nullptr.
	LEAN_INLINE ID3D11ShaderResourceView* GetTexture() const { return m_pTexture; }
	/// Gets the description.
	LEAN_INLINE const DX11::TextureTargetDesc& GetDesc() const { return m_desc; }

	/// Gets the n-th texture.
	LEAN_INLINE ID3D11ShaderResourceView* GetTexture(uint4 n) const { return (m_pTextures) ? m_pTextures[n] : m_pTexture; }

	/// Resets the texture's use count.
	LEAN_INLINE void ResetUses() { m_uses = m_references; }
	/// Returns true, iff the texture has been in use at least once.
	LEAN_INLINE bool WasUsed() { return (m_uses != 0); }

	/// Marks the texture as in use.
	LEAN_INLINE void AddRef() const { ++m_references; ++m_uses; }
	/// Marks the texture as free.
	LEAN_INLINE void Release() const { --m_references; }
	/// Marks the texture as in use.
	LEAN_INLINE bool IsInUse() const { return (m_references != 0); }
};

/// Color texture target.
class ColorTextureTarget : public TextureTarget
{
public:
	typedef lean::com_ptr<ID3D11RenderTargetView> target_ptr;
	typedef lean::scoped_ptr<target_ptr[]> scoped_target_array_ptr;

private:
	lean::com_ptr<ID3D11RenderTargetView> m_pTarget;

	scoped_target_array_ptr m_pTargets;

public:
	/// Constructor. Texture is OPTIONAL.
	BE_GRAPHICS_API ColorTextureTarget(const DX11::TextureTargetDesc &desc, ID3D11Resource *pResource, 
		ID3D11ShaderResourceView *pTexture, ID3D11RenderTargetView *pTarget,
		target_ptr *pTargets = nullptr, texture_ptr *pTextures = nullptr);
	/// Destructor.
	BE_GRAPHICS_API ~ColorTextureTarget();

	/// Gets the target.
	LEAN_INLINE ID3D11RenderTargetView* GetTarget() const { return m_pTarget; }

	/// Gets the n-th target.
	LEAN_INLINE ID3D11RenderTargetView* GetTarget(uint4 n) const { return (m_pTargets) ? m_pTargets[n] : m_pTarget; }
};

/// Depth-stencil texture target.
class DepthStencilTextureTarget : public TextureTarget
{
public:
	typedef lean::com_ptr<ID3D11DepthStencilView> target_ptr;
	typedef lean::scoped_ptr<target_ptr[]> scoped_target_array_ptr;

private:
	lean::com_ptr<ID3D11DepthStencilView> m_pTarget;

	typedef lean::com_ptr<ID3D11DepthStencilView> target_ptr;
	scoped_target_array_ptr m_pTargets;

public:
	/// Constructor. Texture is OPTIONAL.
	BE_GRAPHICS_API DepthStencilTextureTarget(const DX11::TextureTargetDesc &desc, ID3D11Resource *pResource, 
		ID3D11ShaderResourceView *pTexture, ID3D11DepthStencilView *pTarget, target_ptr *pTargets = nullptr);
	/// Destructor.
	BE_GRAPHICS_API ~DepthStencilTextureTarget();

	/// Gets the target.
	LEAN_INLINE ID3D11DepthStencilView* GetTarget() const { return m_pTarget; }

	/// Gets the n-th target.
	LEAN_INLINE ID3D11DepthStencilView* GetTarget(uint4 n) const { return (m_pTargets) ? m_pTargets[n] : m_pTarget; }
};

/// Depth-stencil texture target.
class StageTextureTarget : public TextureTarget
{
public:
	/// Constructor. Texture is OPTIONAL.
	BE_GRAPHICS_API StageTextureTarget(const DX11::TextureTargetDesc &desc, ID3D11Resource *pResource);
	/// Destructor.
	BE_GRAPHICS_API ~StageTextureTarget();
};

namespace DX11
{

using beGraphics::TextureTarget;
using beGraphics::ColorTextureTarget;
using beGraphics::DepthStencilTextureTarget;
using beGraphics::StageTextureTarget;

/// Texture interface.
class TextureTargetPool : public beGraphics::TextureTargetPool
{
private:
	struct M;
	lean::pimpl_ptr<M> m;

public:
	/// Constructor.
	BE_GRAPHICS_API TextureTargetPool(ID3D11Device *pDevice);
	/// Destructor.
	BE_GRAPHICS_API ~TextureTargetPool();

	/// Resets the usage statistics.
	BE_GRAPHICS_API void ResetUsage();
	/// Releases unused targets.
	BE_GRAPHICS_API void ReleaseUnused();

	/// Gets the implementation identifier.
	ImplementationID GetImplementationID() const { return DX11Implementation; };

	/// Acquires a color texture target according to the given description.
	lean::com_ptr<const ColorTextureTarget, true> AcquireColorTextureTarget(const beGraphics::TextureTargetDesc &desc)
	{
		return AcquireColorTextureTarget( ToAPI(desc) );
	}
	/// Acquires a depth-stencil texture target according to the given description.
	lean::com_ptr<const DepthStencilTextureTarget, true> AcquireDepthStencilTextureTarget(const beGraphics::TextureTargetDesc &desc)
	{
		return AcquireDepthStencilTextureTarget( ToAPI(desc) );
	}
	/// Acquires a stage texture target according to the given description.
	lean::com_ptr<const StageTextureTarget, true> AcquireStageTextureTarget(const beGraphics::TextureTargetDesc &desc)
	{
		return AcquireStageTextureTarget( ToAPI(desc) );
	}

	/// Acquires a color texture target according to the given description.
	BE_GRAPHICS_API lean::com_ptr<const ColorTextureTarget, true> AcquireColorTextureTarget(const TextureTargetDesc &desc);
	/// Acquires a depth-stencil texture target according to the given description.
	BE_GRAPHICS_API lean::com_ptr<const DepthStencilTextureTarget, true> AcquireDepthStencilTextureTarget(const TextureTargetDesc &desc);
	/// Acquires a stage texture target according to the given description.
	BE_GRAPHICS_API lean::com_ptr<const StageTextureTarget, true> AcquireStageTextureTarget(const TextureTargetDesc &desc);

	/// Schedules the given subresource for read back.
	BE_GRAPHICS_API lean::com_ptr<const StageTextureTarget, true> ScheduleReadback(const ColorTextureTarget *pTarget, const beGraphics::DeviceContext &context, uint4 subResource = 0);
	/// Reads back from the given stage target.
	BE_GRAPHICS_API bool ReadBack(const StageTextureTarget *pTarget, void *memory, uint4 size, const beGraphics::DeviceContext &context);
	/// Reads back color target information.
	BE_GRAPHICS_API bool ReadBack(const ColorTextureTarget *pTarget, void *memory, uint4 size, const beGraphics::DeviceContext &context, uint4 subResource = 0);
};

template <> struct ToImplementationDX11<beGraphics::TextureTargetPool> { typedef TextureTargetPool Type; };

} // namespace

} // namespace

#endif