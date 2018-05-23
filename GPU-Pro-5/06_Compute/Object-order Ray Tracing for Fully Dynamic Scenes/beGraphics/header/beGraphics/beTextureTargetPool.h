/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_TEXTURE_TARGET_POOL
#define BE_GRAPHICS_TEXTURE_TARGET_POOL

#include "beGraphics.h"
#include "beFormat.h"
#include "beDeviceContext.h"
#include <lean/tags/noncopyable.h>
#include <lean/smart/com_ptr.h>
#include <beCore/beShared.h>

namespace beGraphics
{

/// Texture target flags enumeration.
struct TextureTargetFlags
{
	enum T
	{
		AutoGenMipMaps = (uint4(1) << 31)	///< Use built-in mip map generation.
	};
	LEAN_MAKE_ENUM_STRUCT(TextureTargetFlags)
};

/// Texture target description.
struct TextureTargetDesc
{
	uint4 Width;				///< Target width.
	uint4 Height;				///< Target height.
	uint4 MipLevels;			///< Mip level count. Set most significant bit for auto-generation.
	Format::T Format;			///< Pixel format.
	SampleDesc Samples;			///< Multi-sampling options.
	uint4 Count;				///< Number of texture array elements.

	/// NON-INITIALIZING constructor.
	TextureTargetDesc() { }
	/// Constructor.
	TextureTargetDesc(uint4 width, uint4 height,
		uint4 mipLevels,
		Format::T format,
		const SampleDesc &samples,
		uint4 count = 1)
			: Width(width),
			Height(height),
			MipLevels(mipLevels),
			Format(format),
			Samples(samples),
			Count(count) { }
};

/// Texture target.
class TextureTarget;
/// Color texture target.
class ColorTextureTarget;
/// Depth-stencil texture target.
class DepthStencilTextureTarget;
/// Stage texture target.
class StageTextureTarget;

/// Texture interface.
class TextureTargetPool : public lean::noncopyable_chain<beCore::Resource>, public Implementation
{
public:
	virtual ~TextureTargetPool() throw() { }

	/// Resets the usage statistics.
	virtual void ResetUsage() = 0;
	/// Releases unused targets.
	virtual void ReleaseUnused() = 0;

	/// Acquires a color texture target according to the given description.
	virtual lean::com_ptr<const ColorTextureTarget, true> AcquireColorTextureTarget(const TextureTargetDesc &desc) = 0;
	/// Acquires a depth-stencil texture target according to the given description.
	virtual lean::com_ptr<const DepthStencilTextureTarget, true> AcquireDepthStencilTextureTarget(const TextureTargetDesc &desc) = 0;
	/// Acquires a stage texture target according to the given description.
	virtual lean::com_ptr<const StageTextureTarget, true> AcquireStageTextureTarget(const TextureTargetDesc &desc) = 0;

	/// Schedules the given subresource for read back.
	virtual lean::com_ptr<const StageTextureTarget, true> ScheduleReadback(const ColorTextureTarget *pTarget, const DeviceContext &context, uint4 subResource = 0) = 0;
	/// Reads back from the given stage target.
	virtual bool ReadBack(const StageTextureTarget *pTarget, void *memory, uint4 size, const DeviceContext &context) = 0;
	/// Reads back color target information.
	virtual bool ReadBack(const ColorTextureTarget *pTarget, void *memory, uint4 size, const DeviceContext &context, uint4 subResource = 0) = 0;
};

} // namespace

#endif