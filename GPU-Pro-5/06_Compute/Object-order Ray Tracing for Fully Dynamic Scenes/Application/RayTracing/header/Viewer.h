#pragma once

#include "Tracing.h"

#include <beScene/beCameraController.h>
#include <beScene/beEffectDrivenRenderer.h>
#include <beScene/beResourceManager.h>

#include <lean/smart/resource_ptr.h>

namespace app
{

/// Viewer class.
class Viewer
{
private:
	lean::resource_ptr<besc::CameraController> m_camera;

public:
	/// Constructor.
	Viewer(besc::CameraController *camera, besc::EffectDrivenRenderer &renderer, besc::ResourceManager &resourceManager);
	/// Destructor.
	~Viewer();

	/// Steps the map.
	void Step(float timeStep);
};

/// Sets up rendering to the back buffer.
void SetUpBackBufferRendering(besc::CameraController &camera, besc::EffectDrivenRenderer &renderer);

} // namespace