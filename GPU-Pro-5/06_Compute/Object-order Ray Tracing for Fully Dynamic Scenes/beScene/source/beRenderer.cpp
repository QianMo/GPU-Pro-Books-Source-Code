/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beRenderer.h"
#include "beScene/bePipePool.h"
#include "beScene/bePerspectivePool.h"
#include <beGraphics/Any/beDevice.h>
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beTextureTargetPool.h>
#include "beScene/beRenderingPipeline.h"

namespace beScene
{

namespace
{

/// Creates an immediate device context wrapper.
beGraphics::DeviceContext* CreateImmediateContext(beGraphics::Device *device)
{
	lean::com_ptr<ID3D11DeviceContext> context;
	ToImpl(*device)->GetImmediateContext(context.rebind());
	return new beGraphics::Any::DeviceContext(context);
}

/// Clones a device context wrapper.
beGraphics::DeviceContext* CloneContext(const beGraphics::DeviceContext &context)
{
	return new beGraphics::Any::DeviceContext( ToImpl(context) );
}

// Creates a target pool from the given device.
lean::resource_ptr<beGraphics::TextureTargetPool, true> CreateTargetPool(const beGraphics::Device &device)
{
	return new_resource beGraphics::Any::TextureTargetPool(ToImpl(device));
}

// Creates a pipe pool from the given device.
lean::resource_ptr<PipePool, true> CreatePipePool(beGraphics::TextureTargetPool *pTargetPool)
{
	return new_resource PipePool(pTargetPool);
}

// Creates a perspective pool from the given device.
lean::resource_ptr<PerspectivePool, true> CreatePerspectivePool(PipePool *pPipePool)
{
	return new_resource PerspectivePool(pPipePool);
}

// Creates a rendering pipeline.
lean::resource_ptr<RenderingPipeline, true> CreatePipeline()
{
	return new_resource RenderingPipeline("Renderer.Pipeline");
}

} // namespace

// Constructor.

// Constructor.
Renderer::Renderer(beGraphics::Device *device, beGraphics::TextureTargetPool *targetPool,
	class PipePool *pipePool, class PerspectivePool *perspectivePool, beScene::RenderingPipeline *pipeline)
	: Device( LEAN_ASSERT_NOT_NULL(device) ),
	ImmediateContext( CreateImmediateContext(device) ),
	TargetPool( LEAN_ASSERT_NOT_NULL(targetPool) ),
	PipePool( LEAN_ASSERT_NOT_NULL(pipePool) ),
	PerspectivePool( LEAN_ASSERT_NOT_NULL(perspectivePool) ),
	Pipeline( LEAN_ASSERT_NOT_NULL(pipeline) )
{
}

// Copy constructor.
Renderer::Renderer(const Renderer &right)
	// MONITOR: REDUNDANT
	: Device( LEAN_ASSERT_NOT_NULL(right.Device) ),
	ImmediateContext( CloneContext(*right.ImmediateContext) ),
	TargetPool( LEAN_ASSERT_NOT_NULL(right.TargetPool) ),
	PipePool( LEAN_ASSERT_NOT_NULL(right.PipePool) ),
	PerspectivePool( LEAN_ASSERT_NOT_NULL(right.PerspectivePool) ),
	Pipeline( LEAN_ASSERT_NOT_NULL(right.Pipeline) )
{
}

// Destructor.
Renderer::~Renderer()
{
}

// Commits / reacts to changes.
void Renderer::Commit()
{
}

// Invalidates all caches.
void Renderer::InvalidateCaches()
{
}

// Creates a renderer from the given device.
lean::resource_ptr<Renderer, true> CreateRenderer(beGraphics::Device *device)
{
	return CreateRenderer(device, nullptr);
}

// Creates a renderer from the given device & pipeline.
lean::resource_ptr<Renderer, true> CreateRenderer(beGraphics::Device *device, beScene::RenderingPipeline *pPipeline)
{
	return CreateRenderer(device, nullptr, nullptr, nullptr, pPipeline);
}

// Creates a renderer from the given device, target pool & pipeline.
lean::resource_ptr<Renderer, true> CreateRenderer(beGraphics::Device *device, beGraphics::TextureTargetPool *pTargetPool,
	PipePool *pPipePool, PerspectivePool *pPerspectivePool, beScene::RenderingPipeline *pPipeline)
{
	LEAN_ASSERT(device != nullptr);

	lean::resource_ptr<beg::TextureTargetPool> targetPool = (pTargetPool) ? lean::secure_resource(pTargetPool) : CreateTargetPool(*device);
	lean::resource_ptr<PipePool> pipePool = (pPipePool) ? lean::secure_resource(pPipePool) : CreatePipePool(targetPool);
	lean::resource_ptr<PerspectivePool> perspectivePool = (pPerspectivePool) ? lean::secure_resource(pPerspectivePool) : CreatePerspectivePool(pipePool);
	lean::resource_ptr<RenderingPipeline> pipeline = (pPipeline) ? lean::secure_resource(pPipeline) : CreatePipeline();

	return new_resource Renderer(device, targetPool, pipePool, perspectivePool, pipeline);
}

} // namespace
