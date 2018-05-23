/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERINGCONTROLLER
#define BE_SCENE_RENDERINGCONTROLLER

#include "beScene.h"
#include <beEntitySystem/beSimulationController.h>
#include <beEntitySystem/beSynchronizedHost.h>
#include <beEntitySystem/beAnimatedHost.h>
#include <beEntitySystem/beRenderable.h>
#include "beScene/bePerspectiveHost.h"
#include "beScene/beRenderableHost.h"
#include <lean/smart/resource_ptr.h>

#include "beScene/beRenderingLimits.h"

namespace beScene
{

// Prototypes
class RenderContext;
class RenderingPipeline;
class PipelinePerspective;
class Caching;

/// Scene controller.
class RenderingController : public beEntitySystem::SimulationController,
	public beEntitySystem::SynchronizedHost, public beEntitySystem::AnimatedHost, public beEntitySystem::Renderable,
	public RenderableHost, public PerspectiveHost
{
private:
	lean::resource_ptr<RenderingPipeline> m_renderingPipeline;
	lean::resource_ptr<RenderContext> m_pRenderContext;
	Caching *m_pCaching;
	
	beEntitySystem::Simulation *m_pAttachedTo;

public:
	/// Constructor.
	BE_SCENE_API RenderingController(RenderingPipeline *renderingPipeline, RenderContext *pRenderContext = nullptr);
	/// Destructor.
	BE_SCENE_API ~RenderingController();

	/// Synchronizes the simulation with this controller.
	BE_SCENE_API void Fetch();

	/// Renders the scene using the stored context.
	BE_SCENE_API void Render();
	/// Renders the scene using the given context.
	BE_SCENE_API void Render(RenderContext &renderContext, PipelineStageMask overrideStageMask = 0);
	/// Renders the scene using the given context.
	BE_SCENE_API void Render(PipelinePerspective &perspective, RenderContext &renderContext, PipelineStageMask overrideStageMask = 0);

	/// Sets the render context.
	BE_SCENE_API void SetRenderContext(RenderContext *pRenderContext);
	/// Gets the render context.
	LEAN_INLINE RenderContext* GetRenderContext() const { return m_pRenderContext; }

	/// Attaches this controller to its simulation.
	BE_SCENE_API void Attach(beEntitySystem::Simulation *simulation);
	/// Detaches this controller from its simulation.
	BE_SCENE_API void Detach(beEntitySystem::Simulation *simulation);

	/// Gets the pipeline.
	LEAN_INLINE RenderingPipeline* GetRenderingPipeline() { return m_renderingPipeline; }
	/// Gets the pipeliney.
	LEAN_INLINE const RenderingPipeline* GetRenderingPipeline() const { return m_renderingPipeline; }

	/// Sets a caching object to be invalidated on flush.
	LEAN_INLINE void SetCaching(Caching *pCaching) { m_pCaching = pCaching; }
	/// Sets a caching object to be invalidated on flush.
	LEAN_INLINE Caching* GetCaching() const { return m_pCaching; }

	/// Gets the controller type.
	BE_SCENE_API static const beCore::ComponentType* GetComponentType();
	/// Gets the controller type.
	BE_SCENE_API const beCore::ComponentType* GetType() const;
};

} // nmaespace

#endif