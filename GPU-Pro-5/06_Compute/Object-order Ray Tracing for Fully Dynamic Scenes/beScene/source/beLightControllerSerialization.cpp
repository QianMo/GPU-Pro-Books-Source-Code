/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include <beEntitySystem/beGenericGroupControllerSerializer.h>
#include "beScene/beLightControllers.h"
#include "beScene/beRenderingController.h"

#include <beEntitySystem/beWorld.h>
#include <beEntitySystem/beSerialization.h>

#include <beEntitySystem/beSerializationParameters.h>
#include "beScene/beSerializationParameters.h"
#include <beCore/beParameterSet.h>
#include <beCore/beParameters.h>

#include "beScene/beLightMaterial.h"
#include "beScene/beLightMaterialCache.h"
#include "beScene/beInlineMaterialSerialization.h"

#include "beScene/beResourceManager.h"
#include "beScene/beEffectDrivenRenderer.h"

#include <lean/xml/numeric.h>
#include <lean/logging/errors.h>

namespace beScene
{

/// Serializes mesh controllers.
template <class LightController>
class LightControllerSerializer : public bees::GenericGroupControllerSerializer<
	LightController,
	LightControllers<LightController>,
	LightControllerSerializer<LightController>
>
{
public:
	typedef LightControllers<LightController> LightControllers;

	LightControllers* CreateControllerGroup(bees::World &world, const bec::ParameterSet &parameters) const
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);
		RenderingController *renderingCtrl = LEAN_THROW_NULL(sceneParameters.RenderingController);

		lean::scoped_ptr<LightControllers> lights = CreateLightControllers<LightController>(
				&world.PersistentIDs(),
				sceneParameters.Renderer->PerspectivePool(), *sceneParameters.Renderer->Pipeline(),
				*sceneParameters.Renderer->Device()
			);
		lights->SetComponentMonitor(sceneParameters.ResourceManager->Monitor());
		renderingCtrl->AddRenderable(lights);

		return lights.detach();
	}

	// Creates a serializable object from the given parameters.
	lean::scoped_ptr<beEntitySystem::Controller, lean::critical_ref> Create(
		const beCore::Parameters &creationParameters, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);
		LightControllers* lightControllers = GetControllerGroup(parameters);
		
		// Light controller
		lean::scoped_ptr<LightController> controller( lightControllers->AddController() );
		controller->SetMaterial( GetLightDefaultMaterial<LightController>(*sceneParameters.ResourceManager, *sceneParameters.Renderer) );

		return controller.transfer();
	}

	// Loads a mesh controller from the given xml node.
	lean::scoped_ptr<beEntitySystem::Controller, lean::critical_ref> Load(const rapidxml::xml_node<lean::utf8_t> &node,
		beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);
		LightControllers* lightControllers = GetControllerGroup(parameters);

		lean::resource_ptr<LightMaterial> lightMaterial;

		utf8_ntr materialName = lean::get_attribute(node, "material");

		if (!materialName.empty())
			lightMaterial = sceneParameters.Renderer->LightMaterials->GetMaterial(
					sceneParameters.ResourceManager->MaterialCache->GetByName(materialName, true)
				);
		else
			lightMaterial = GetLightDefaultMaterial<LightController>(*sceneParameters.ResourceManager, *sceneParameters.Renderer);

		lean::scoped_ptr<LightController> controller( lightControllers->AddController() );
		controller->SetMaterial(lightMaterial);
		ControllerSerializer::Load(controller, node, parameters, queue);

		return controller.transfer();
	}

	// Saves the given mesh controller to the given XML node.
	void Save(const beEntitySystem::Controller *pSerializable, rapidxml::xml_node<lean::utf8_t> &node,
		beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue) const LEAN_OVERRIDE
	{
		ControllerSerializer::Save(pSerializable, node, parameters, queue);

		const LightController &lightController = static_cast<const LightController&>(*pSerializable);
	
		if (const LightMaterial *lightMaterial = lightController.GetMaterial())
		{
			utf8_ntr name = bec::GetCachedName<utf8_ntr>(lightMaterial->GetMaterial());
			if (!name.empty())
				lean::append_attribute( *node.document(), node, "material", name );
			else
				LEAN_LOG_ERROR_MSG("Could not identify LightController material, information will be lost");

			SaveMaterial(lightMaterial->GetMaterial(), parameters, queue);
		}
	}
};

const beEntitySystem::EntityControllerSerializationPlugin< LightControllerSerializer<DirectionalLightController> > DirectionalLightControllerSerialization;
const beEntitySystem::EntityControllerSerializationPlugin< LightControllerSerializer<PointLightController> > PointLightControllerSerialization;
const beEntitySystem::EntityControllerSerializationPlugin< LightControllerSerializer<SpotLightController> > SpotLightControllerSerialization;

} // namespace
