/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERER
#define BE_SCENE_RENDERER

#include "beScene.h"
#include "beCaching.h"
#include <beCore/beShared.h>
#include <beGraphics/beDevice.h>
#include <beGraphics/beDeviceContext.h>
#include <beGraphics/beTextureTargetPool.h>
#include <lean/smart/resource_ptr.h>
#include <lean/smart/scoped_ptr.h>

namespace beScene
{
	
// Prototypes
class RenderingPipeline;
class PipePool;
class PerspectivePool;

/// Renderer.
class Renderer : public beCore::Resource, public Caching
{
public:
	/// Constructor.
	BE_SCENE_API Renderer(beGraphics::Device *pDevice, beGraphics::TextureTargetPool *pTargetPool,
		PipePool *pPipePool, PerspectivePool *pPerspectivePool, beScene::RenderingPipeline *pPipeline);
	/// Copy constructor.
	BE_SCENE_API Renderer(const Renderer &right);
	/// Destructor.
	BE_SCENE_API virtual ~Renderer();

	/// Invalidates all caches.
	BE_SCENE_API virtual void InvalidateCaches();

	/// Device.
	lean::resource_ptr<beGraphics::Device> Device;
	/// Immediate device context.
	lean::scoped_ptr<beGraphics::DeviceContext> ImmediateContext;

	/// Texture target pool.
	lean::resource_ptr<beGraphics::TextureTargetPool> TargetPool;
	/// Pipe pool.
	lean::resource_ptr<PipePool> PipePool;

	/// Perspective Pool.
	lean::resource_ptr<PerspectivePool> PerspectivePool;
	/// Pipeline.
	lean::resource_ptr<RenderingPipeline> Pipeline;

	/// Commits / reacts to changes.
	BE_SCENE_API virtual void Commit();
};

/// Creates a renderer from the given device.
BE_SCENE_API lean::resource_ptr<Renderer, true> CreateRenderer(beGraphics::Device *device);
/// Creates a renderer from the given device & pipeline.
BE_SCENE_API lean::resource_ptr<Renderer, true> CreateRenderer(beGraphics::Device *device, beScene::RenderingPipeline *pPipeline);
/// Creates a renderer from the given device, target pool & pipeline.
BE_SCENE_API lean::resource_ptr<Renderer, true> CreateRenderer(beGraphics::Device *device, beGraphics::TextureTargetPool *pTargetPool,
	PipePool *pPipePool, PerspectivePool *pPerspectivePool, beScene::RenderingPipeline *pPipeline);

} // namespace

#endif