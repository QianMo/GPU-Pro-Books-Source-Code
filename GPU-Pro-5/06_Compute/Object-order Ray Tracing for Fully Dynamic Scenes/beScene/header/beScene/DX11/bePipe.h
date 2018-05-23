/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PIPE_DX11
#define BE_SCENE_PIPE_DX11

#include "../beScene.h"
#include "../bePipe.h"
#include <beGraphics/Any/beTextureTargetPool.h>
#include <beGraphics/Any/beTexture.h>
#include <lean/smart/resource_ptr.h>
#include <vector>
#include <memory>

namespace beScene
{

namespace DX11
{

/// Rendering perspective.
class Pipe : public beScene::Pipe
{
public:
	struct ColorTarget;
	typedef std::vector<ColorTarget> color_target_vector;
	struct DepthStencilTarget;
	typedef std::vector<DepthStencilTarget> depth_stencil_target_vector;

	struct FinalTarget;

private:
	lean::pimpl_ptr<beGraphics::Any::TextureTargetDesc> m_desc;

	beGraphics::Any::TextureTargetPool *m_pTargetPool;

	// ORDER: referenced by target vectors!
	std::auto_ptr<beGraphics::Any::ColorTextureTarget> m_pFinalTarget;
	beGraphics::Viewport m_viewport;

	color_target_vector m_colorTargets;
	depth_stencil_target_vector m_depthStencilTargets;

	bool m_bKeepResults;

public:
	/// Constructor.
	BE_SCENE_API Pipe(const beGraphics::Any::Texture &finalTarget, beGraphics::Any::TextureTargetPool *pTargetPool);
	/// Constructor.
	BE_SCENE_API Pipe(const beGraphics::Any::TextureTargetDesc &desc, beGraphics::Any::TextureTargetPool *pTargetPool);
	/// Destructor.
	BE_SCENE_API ~Pipe();

	/// Gets a new color target matching the given description.
	BE_SCENE_API lean::com_ptr<const beGraphics::Any::ColorTextureTarget, true> NewColorTarget(const beGraphics::Any::TextureTargetDesc &desc, uint4 flags) const;
	/// Gets a new depth-stencil target matching the given description.
	BE_SCENE_API lean::com_ptr<const beGraphics::Any::DepthStencilTextureTarget, true> NewDepthStencilTarget(const beGraphics::Any::TextureTargetDesc &desc, uint4 flags) const;


	/// Gets the target identified by the given name or nullptr if none available.
	BE_SCENE_API const beGraphics::Any::TextureTarget* GetAnyTarget(const utf8_ntri &name) const;
	/// Gets the color target identified by the given name or nullptr if none available.
	BE_SCENE_API const beGraphics::Any::ColorTextureTarget* GetColorTarget(const utf8_ntri &name) const;
	/// Gets the depth-stencil target identified by the given name nullptr if none available.
	BE_SCENE_API const beGraphics::Any::DepthStencilTextureTarget* GetDepthStencilTarget(const utf8_ntri &name) const;

	/// Updates the color target identified by the given name.
	BE_SCENE_API void SetColorTarget(const utf8_ntri &name,
		const beGraphics::Any::ColorTextureTarget *pTarget, uint4 flags, uint4 outputIndex,
		lean::com_ptr<const beGraphics::Any::ColorTextureTarget> *pOldTarget = nullptr);
	/// Updates the color target identified by the given name.
	BE_SCENE_API void SetDepthStencilTarget(const utf8_ntri &name,
		const beGraphics::Any::DepthStencilTextureTarget *pTarget, uint4 flags, uint4 outputIndex,
		lean::com_ptr<const beGraphics::Any::DepthStencilTextureTarget> *pOldTarget = nullptr);
	

	/// Gets a new color target matching the given description and stores it under the given name.
	BE_SCENE_API const beGraphics::Any::ColorTextureTarget* GetNewColorTarget(const utf8_ntri &name,
		const beGraphics::Any::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex,
		lean::com_ptr<const beGraphics::Any::ColorTextureTarget> *pOldTarget = nullptr);
	/// Gets a new depth-stencil target matching the given description and stores it under the given name.
	BE_SCENE_API const beGraphics::Any::DepthStencilTextureTarget* GetNewDepthStencilTarget(const utf8_ntri &name,
		const beGraphics::Any::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex,
		lean::com_ptr<const beGraphics::Any::DepthStencilTextureTarget> *pOldTarget = nullptr);

	/// Gets the color target identified by the given name or adds one according to the given description.
	BE_SCENE_API const beGraphics::Any::ColorTextureTarget* GetColorTarget(const utf8_ntri &name,
		const beGraphics::Any::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, bool *pIsNew = nullptr);
	/// Gets the depth-stencil target identified by the given name or adds one according to the given description.
	BE_SCENE_API const beGraphics::Any::DepthStencilTextureTarget* GetDepthStencilTarget(const utf8_ntri &name,
		const beGraphics::Any::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, bool *pIsNew = nullptr);

	/// Resets all pipe contents.
	BE_SCENE_API void Reset(const beGraphics::Any::TextureTargetDesc &desc);
	/// Release all non-permanent pipe contents.
	BE_SCENE_API void Release();

	/// Instructs the pipe to keep its results on release.
	BE_SCENE_API void KeepResults(bool bKeep = true);

	/// (Re)sets the final target.
	BE_SCENE_API void SetFinalTarget(const beGraphics::Any::Texture *pFinalTarget);
	/// Gets the target identified by the given name or nullptr if none available.
	BE_SCENE_API const beGraphics::Any::TextureTarget* GetFinalTarget() const;

	/// Sets a viewport.
	BE_SCENE_API void SetViewport(const beGraphics::Viewport &viewport);
	/// Gets the current viewport.
	BE_SCENE_API const beGraphics::Viewport& GetViewport() const { return m_viewport; }

	/// (Re)sets the description.
	BE_SCENE_API void SetDesc(const beGraphics::Any::TextureTargetDesc &desc);
	/// Gets the description.
	LEAN_INLINE const beGraphics::Any::TextureTargetDesc& GetDesc() const { return *m_desc; }

	/// Gets the implementation identifier.
	LEAN_INLINE beGraphics::ImplementationID GetImplementationID() const { return beGraphics::DX11Implementation; }
};

} // namespace

using beGraphics::DX11::ToImpl;

} // namespace

namespace beGraphics
{
	namespace DX11
	{
		template <> struct ToImplementationDX11<beScene::Pipe> { typedef beScene::DX11::Pipe Type; };
	} // namespace
} // namespace

#endif