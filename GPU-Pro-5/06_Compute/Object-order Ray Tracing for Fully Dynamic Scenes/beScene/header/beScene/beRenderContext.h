/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDER_CONTEXT
#define BE_SCENE_RENDER_CONTEXT

#include "beScene.h"
#include <beCore/beShared.h>
#include <beGraphics/beDeviceContext.h>
#include <beGraphics/beStateManager.h>
#include <lean/smart/resource_ptr.h>
#include <lean/smart/scoped_ptr.h>

namespace beScene
{

/// Render context.
class RenderContext : public beCore::Resource
{
private:
	const lean::scoped_ptr<beGraphics::DeviceContext> m_pContext;

	const lean::resource_ptr<beGraphics::StateManager> m_pStateManager;

public:
	/// Constructor.
	BE_SCENE_API RenderContext(const beGraphics::DeviceContext &context, beGraphics::StateManager *pStateManager);
	/// Copy constructor.
	BE_SCENE_API RenderContext(const RenderContext &right);
	/// Destructor.
	BE_SCENE_API ~RenderContext();

	/// Gets the immediate device context.
	LEAN_INLINE beGraphics::DeviceContext& Context() const { return *m_pContext; }

	/// Gets the state manager.
	LEAN_INLINE beGraphics::StateManager& StateManager() const { return *m_pStateManager; }
};

/// Creates a render context from the given device context.
BE_SCENE_API lean::resource_ptr<RenderContext, true> CreateRenderContext(const beGraphics::DeviceContext &context);
/// Creates a render context from the given device context.
BE_SCENE_API lean::resource_ptr<RenderContext, true> CreateRenderContext(const beGraphics::DeviceContext &context, beGraphics::StateManager *pStateManager);

} // namespace

#endif