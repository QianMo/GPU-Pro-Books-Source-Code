/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beRenderingController.h"
#include <beEntitySystem/beSimulation.h>
#include "beScene/beRenderingPipeline.h"
#include "beScene/beRenderContext.h"
#include "beScene/beCaching.h"

#include <lean/logging/errors.h>

namespace beScene
{

BE_CORE_PUBLISH_COMPONENT(RenderingController)

// Constructor.
RenderingController::RenderingController(RenderingPipeline *renderingPipeline, RenderContext *pRenderContext)
	: m_renderingPipeline( LEAN_ASSERT_NOT_NULL(renderingPipeline) ),
	m_pRenderContext(pRenderContext),
	m_pCaching( nullptr ),

	m_pAttachedTo( nullptr )
{
}

// Destructor.
RenderingController::~RenderingController()
{
}

// Synchronizes the simulation with this controller.
void RenderingController::Fetch()
{
	if (m_pCaching)
		m_pCaching->InvalidateCaches();

	SynchronizedHost::Fetch();
}

// Renders the scene using the stored context.
void RenderingController::Render()
{
	if (m_pRenderContext)
		Render(*m_pRenderContext);
	else
		LEAN_LOG_ERROR_MSG("Cannot render without a valid render context set.");
}

// Renders the scene using the given context.
void RenderingController::Render(RenderContext &renderContext, PipelineStageMask overrideStageMask)
{
	const Renderables renderables = GetRenderables();

	for (Perspectives perspectives = GetPerspectives(); perspectives.Begin < perspectives.End; ++perspectives.Begin)
		Render(**perspectives.Begin, renderContext, overrideStageMask);
}

// Renders the scene using the given context.
void RenderingController::Render(PipelinePerspective &perspective, RenderContext &renderContext, PipelineStageMask overrideStageMask)
{
	const Renderables renderables = GetRenderables();
	
	m_renderingPipeline->Prepare(perspective, renderables.Begin, Size4(renderables), overrideStageMask);
	m_renderingPipeline->Optimize(perspective, renderables.Begin, Size4(renderables), overrideStageMask);
	m_renderingPipeline->Render(perspective, renderables.Begin, Size4(renderables), renderContext, overrideStageMask);
	m_renderingPipeline->ReleaseIntermediate(perspective, renderables.Begin, Size4(renderables));
}

// Sets the render context.
void RenderingController::SetRenderContext(RenderContext *pRenderContext)
{
	m_pRenderContext = pRenderContext;
}

// Attaches this controller to its simulation.
void RenderingController::Attach(beEntitySystem::Simulation *simulation)
{
	if (m_pAttachedTo)
	{
		LEAN_LOG_ERROR_MSG("rendering controller already attached to simulation");
		return;
	}

	// ORDER: Active as soon as SOMETHING MIGHT have been attached
	m_pAttachedTo = LEAN_ASSERT_NOT_NULL(simulation);

	simulation->AddSynchronized(this, beEntitySystem::SynchronizedFlags::All);
	simulation->AddAnimated(this);
	simulation->AddRenderable(this);
}

// Detaches this controller from its simulation.
void RenderingController::Detach(beEntitySystem::Simulation *simulation)
{
	if (LEAN_ASSERT_NOT_NULL(simulation) != m_pAttachedTo)
	{
		LEAN_LOG_ERROR_MSG("rendering controller was never attached to simulation");
		return;
	}

	simulation->RemoveSynchronized(this, beEntitySystem::SynchronizedFlags::All);
	simulation->RemoveAnimated(this);
	simulation->RemoveRenderable(this);

	// ORDER: Active as long as ANYTHING MIGHT be attached
	m_pAttachedTo = nullptr;
}

} // namespace
