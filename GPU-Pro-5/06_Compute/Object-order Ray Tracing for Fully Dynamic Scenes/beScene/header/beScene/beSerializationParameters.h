/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_SERIALIZATION_PARAMETERS
#define BE_SCENE_SERIALIZATION_PARAMETERS

#include "beScene.h"

#include <beCore/beParameterSet.h>

namespace beScene
{

// Prototypes
class ResourceManager;
class EffectDrivenRenderer;
class RenderingController;

/// Scene parameter IDs.
struct SceneParameterIDs
{
	uint4 ResourceManager;
	uint4 Renderer;
	uint4 RenderingController;

	/// Non-initializing constructor.
	SceneParameterIDs() { }
	/// Constructor.
	SceneParameterIDs(uint4 resourceManagerID, uint4 rendererID, uint4 renderingControllerID)
			: ResourceManager(resourceManagerID),
			Renderer(rendererID),
			RenderingController(renderingControllerID) { }
};

/// Scene parameters.
struct SceneParameters
{
	ResourceManager *ResourceManager;
	EffectDrivenRenderer *Renderer;
	RenderingController *RenderingController;

	/// Default constructor.
	SceneParameters()
		: ResourceManager(),
		Renderer(),
		RenderingController() { }
	/// Constructor.
	SceneParameters(class ResourceManager *pResourceManager,
		EffectDrivenRenderer *pRenderer,
		class RenderingController *pRenderingController = nullptr)
			: ResourceManager(pResourceManager),
			Renderer(pRenderer),
			RenderingController(pRenderingController) { }
};

/// Gets the serialization parameter IDs.
BE_SCENE_API const SceneParameterIDs& GetSceneParameterIDs();

/// Sets the given scene parameters in the given parameter set.
BE_SCENE_API void SetSceneParameters(beCore::ParameterSet &parameters, const SceneParameters &sceneParameters);
/// Sets the given scene parameters in the given parameter set.
BE_SCENE_API SceneParameters GetSceneParameters(const beCore::ParameterSet &parameters);

} // namespace

#endif