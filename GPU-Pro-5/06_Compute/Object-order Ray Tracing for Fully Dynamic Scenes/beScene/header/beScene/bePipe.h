/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PIPE
#define BE_SCENE_PIPE

#include "beScene.h"
#include <beCore/beShared.h>
#include <beGraphics/beTextureTargetPool.h>
#include <beGraphics/beTexture.h>
#include <beGraphics/beDevice.h>
#include <lean/smart/resource_ptr.h>
#include <vector>
#include <lean/pimpl/pimpl_ptr.h>
#include <memory>

namespace beScene
{

/// Pipe target flags enumeration.
struct PipeTargetFlags
{
	/// Enumeration.
	enum T
	{
		Persistent = 0x1,
		Immutable = 0x10,
		Flash = 0x20,
		Keep = 0x40,

		Output = 0x80
	};
	LEAN_MAKE_ENUM_STRUCT(PipeTargetFlags)
};

/// Maximum number of outputs.
const uint4 MaxPipeOutputCount = 32;
/// Pipe output mask type.
typedef uint4 PipeOutputMask;

LEAN_STATIC_ASSERT(lean::size_info<PipeOutputMask>::bits >= MaxPipeOutputCount);

/// Rendering perspective.
class Pipe : public beCore::Resource, public beGraphics::Implementation
{
public:
	/// Gets a new color target matching the given description.
	BE_SCENE_API lean::com_ptr<const beGraphics::ColorTextureTarget, true> NewColorTarget(const beGraphics::TextureTargetDesc &desc, uint4 flags) const;
	/// Gets a new depth-stencil target matching the given description.
	BE_SCENE_API lean::com_ptr<const beGraphics::DepthStencilTextureTarget, true> NewDepthStencilTarget(const beGraphics::TextureTargetDesc &desc, uint4 flags) const;


	/// Gets the target identified by the given name or nullptr if none available.
	BE_SCENE_API const beGraphics::TextureTarget* GetAnyTarget(const utf8_ntri &name) const;
	/// Gets the color target identified by the given name or nullptr if none available.
	BE_SCENE_API const beGraphics::ColorTextureTarget* GetColorTarget(const utf8_ntri &name) const;
	/// Gets the depth-stencil target identified by the given name nullptr if none available.
	BE_SCENE_API const beGraphics::DepthStencilTextureTarget* GetDepthStencilTarget(const utf8_ntri &name) const;

	/// Updates the color target identified by the given name.
	BE_SCENE_API void SetColorTarget(const utf8_ntri &name,
		const beGraphics::ColorTextureTarget *pTarget, uint4 flags, uint4 outputIndex,
		lean::com_ptr<const beGraphics::ColorTextureTarget> *pOldTarget = nullptr);
	/// Updates the color target identified by the given name.
	BE_SCENE_API void SetDepthStencilTarget(const utf8_ntri &name,
		const beGraphics::DepthStencilTextureTarget *pTarget, uint4 flags, uint4 outputIndex,
		lean::com_ptr<const beGraphics::DepthStencilTextureTarget> *pOldTarget = nullptr);
	

	/// Gets a new color target matching the given description and stores it under the given name.
	BE_SCENE_API const beGraphics::ColorTextureTarget* GetNewColorTarget(const utf8_ntri &name,
		const beGraphics::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex,
		lean::com_ptr<const beGraphics::ColorTextureTarget> *pOldTarget = nullptr);
	/// Gets a new depth-stencil target matching the given description and stores it under the given name.
	BE_SCENE_API const beGraphics::DepthStencilTextureTarget* GetNewDepthStencilTarget(const utf8_ntri &name,
		const beGraphics::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex,
		lean::com_ptr<const beGraphics::DepthStencilTextureTarget> *pOldTarget = nullptr);

	/// Gets the color target identified by the given name or adds one according to the given description.
	BE_SCENE_API const beGraphics::ColorTextureTarget* GetColorTarget(const utf8_ntri &name,
		const beGraphics::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, bool *pIsNew = nullptr);
	/// Gets the depth-stencil target identified by the given name or adds one according to the given description.
	BE_SCENE_API const beGraphics::DepthStencilTextureTarget* GetDepthStencilTarget(const utf8_ntri &name,
		const beGraphics::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, bool *pIsNew = nullptr);

	/// Resets all pipe contents.
	BE_SCENE_API void Reset(const beGraphics::TextureTargetDesc &desc);
	/// Release all non-permanent pipe contents.
	BE_SCENE_API void Release();

	/// Instructs the pipe to keep its results on release.
	BE_SCENE_API void KeepResults(bool bKeep = true);

	/// (Re)sets the final target.
	BE_SCENE_API void SetFinalTarget(const beGraphics::Texture *pFinalTarget);
	/// Gets the target identified by the given name or nullptr if none available.
	BE_SCENE_API const beGraphics::TextureTarget* GetFinalTarget() const;

	/// Sets a viewport.
	BE_SCENE_API void SetViewport(const beGraphics::Viewport &viewport);
	/// Gets the current viewport.
	BE_SCENE_API beGraphics::Viewport GetViewport() const;

	/// (Re)sets the description.
	BE_SCENE_API void SetDesc(const beGraphics::TextureTargetDesc &desc);
	/// Gets the description.
	BE_SCENE_API beGraphics::TextureTargetDesc GetDesc() const;
};

/// Constructor.
BE_SCENE_API lean::resource_ptr<Pipe, true> CreatePipe(const beGraphics::Texture &finalTarget, beGraphics::TextureTargetPool *pTargetPool);
/// Constructor.
BE_SCENE_API lean::resource_ptr<Pipe, true> CreatePipe(const beGraphics::TextureTargetDesc &desc, beGraphics::TextureTargetPool *pTargetPool);

} // namespace

#endif