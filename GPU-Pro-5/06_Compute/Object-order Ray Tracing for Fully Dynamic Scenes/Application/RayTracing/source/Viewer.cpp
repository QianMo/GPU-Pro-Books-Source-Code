/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "Viewer.h"

#include <beGraphics/beMaterial.h>
#include <beScene/beEffectDrivenRenderer.h>
#include <beScene/bePipe.h>
#include <beScene/bePerspectivePool.h>
#include <beScene/bePipelinePerspective.h>
#include <beScene/beProcessingPipeline.h>
#include <beScene/beQuadProcessor.h>
#include <beScene/beRenderingPipeline.h>

namespace app
{

namespace
{
	
/// Adds the given effect to the given processing pipeline
beg::Material* AddEffect(const utf8_ntri &file, const utf8_ntri &args, beScene::ProcessingPipeline &pipeline,
	beScene::EffectDrivenRenderer &renderer, beScene::ResourceManager &resourceManager)
{
	lean::resource_ptr<beGraphics::Effect> processingEffect =
		resourceManager.EffectCache()->GetByFile(file, args, "");

	lean::resource_ptr<beg::Material> processingMaterial = beg::CreateMaterial(
			&processingEffect.get(), 1,
			*resourceManager.EffectCache()
		);

	lean::resource_ptr<beScene::QuadProcessor> pProcessor = new_resource beScene::QuadProcessor(
			renderer.Device(),
			renderer.ProcessingDrivers()
		);
	pProcessor->SetMaterial(processingMaterial);

	pipeline.Add(pProcessor);

	return processingMaterial;
}

} // namespace

// Constructor.
Viewer::Viewer(besc::CameraController *camera, besc::EffectDrivenRenderer &renderer, besc::ResourceManager &resourceManager)
	: m_camera( LEAN_ASSERT_NOT_NULL(camera) )
{
	lean::resource_ptr<besc::ProcessingPipeline> processor = new_resource besc::ProcessingPipeline("Viewer Post Processing");

//	AddEffect("Processing/SSAO.fx", *processor, renderer, resourceManager);
	AddEffect("Processing/SimpleTonemap.fx", "BE_LDR_PROCESSING", *processor, renderer, resourceManager);
	AddEffect("Processing/FXAA.fx", "", *processor, renderer, resourceManager);

	m_camera->SetPerspective( renderer.PerspectivePool()->GetPerspective(nullptr, processor) );
}

// Destructor.
Viewer::~Viewer()
{
}

// Steps the map.
void Viewer::Step(float timeStep)
{
}

// Sets up rendering to the back buffer.
void SetUpBackBufferRendering(besc::CameraController &camera, besc::EffectDrivenRenderer &renderer)
{
	camera.GetPerspective()->SetPipe(
			besc::CreatePipe(
				*beg::GetBackBuffer(*renderer.Device()->GetHeadSwapChain(0)),
				renderer.TargetPool()
			).get()
		);
}

} // namespace